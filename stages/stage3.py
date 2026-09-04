"""Stage 3 — affected assets.

This runs *before* impact analysis for a reason the spec is explicit about:
"positive" is not a statement until you say positive *for what*. A US enforcement
action against an exchange is bearish for that exchange's token, arguably bullish
for its competitors, and roughly neutral for Bitcoin. A stage that produced one
direction for "the market" would be wrong three ways at once.

Unlike Stages 1 and 2 this is not a yes/no vote, so the aggregation is different:
each model returns a list, and an asset survives on how many models named it. That
threshold is a spending control as much as an accuracy control — every surviving
asset costs a full Stage 4 panel, so one model inventing a ticker must not buy
itself five more calls.
"""

from __future__ import annotations

import logging
import re

from services.consensus import Consensus, aggregate_numeric
from services.jsonparse import as_bool, as_probability
from services.models import panel_for_stage
from services.prompts import STAGE3_SYSTEM, stage3_user
from services.types import AssetLink, Relation
from stages.base import PanelRunner, StageContext, best_evidence, majority_of

log = logging.getLogger(__name__)

#: Mentions an asset needs to survive. Two of five is the floor at which a link is
#: someone's finding rather than someone's guess.
ASSET_MIN_VOTES = 2

#: Hard cap, matching the prompt. Beyond this the links are noise.
MAX_ASSETS = 6

#: Tickers are 2-10 characters of letters and digits. This rejects prose that
#: models occasionally put in the asset field ("the whole market", "altcoins").
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")

#: Words that look like tickers but name a category, not an asset.
_NOT_TICKERS = frozenset({
    "ALL", "ANY", "NONE", "MARKET", "CRYPTO", "ALTCOINS", "ALTS", "DEFI",
    "STABLECOINS", "TOKENS", "COINS", "NA", "N/A", "USD", "EUR", "FIAT",
})


def normalize_ticker(value: object) -> str:
    """Coerce a model's asset field to a ticker, or return "" to discard it."""
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text.split("/")[0].split("-")[0])
    if not text or text in _NOT_TICKERS or not _TICKER_RE.match(text):
        return ""
    return text


def parse(data: dict) -> tuple[bool | None, dict]:
    """Read one model's asset list.

    The vote is simply whether the model found any asset at all; the useful output
    is the parsed list in the detail, which :func:`run` aggregates.
    """
    raw = data.get("assets")
    if isinstance(raw, dict):          # a single asset returned unwrapped
        raw = [raw]
    if not isinstance(raw, list):
        raw = []

    parsed: list[dict] = []
    for entry in raw[: MAX_ASSETS * 2]:
        if isinstance(entry, str):
            entry = {"asset": entry}
        if not isinstance(entry, dict):
            continue
        ticker = normalize_ticker(entry.get("asset") or entry.get("symbol") or entry.get("ticker"))
        if not ticker:
            continue
        parsed.append({
            "asset": ticker,
            "relation": Relation.parse(entry.get("relation")).value,
            "confidence": as_probability(entry.get("confidence"), 0.5),
            "evidence": str(entry.get("evidence") or entry.get("reason") or "")[:400],
        })

    detail = {
        "assets": parsed,
        "market_wide": as_bool(data.get("market_wide"), False),
        "reason": ", ".join(item["asset"] for item in parsed) or "no assets named",
    }
    if not parsed and "assets" not in data and "market_wide" not in data:
        return None, detail  # the model did not answer this question at all
    return bool(parsed), detail


async def run(runner: PanelRunner, ctx: StageContext, stage2_reason: str = "") -> Consensus:
    """Identify the assets in the blast radius. Sets ``ctx.links``."""
    panel = panel_for_stage(3)
    consensus, _ = await runner.run(
        ctx,
        stage=3,
        system=STAGE3_SYSTEM,
        user=stage3_user(ctx.news, ctx.facts, stage2_reason),
        parse=parse,
        reject_votes=majority_of(len(panel)),
    )

    ctx.links = aggregate_links(consensus)
    ctx.market_wide = _market_wide(consensus)

    if not ctx.links:
        ctx.links = _fallback_links(ctx)

    if not ctx.links:
        ctx.drop(3, "no affected asset identified")
        log.info("stage 3 found no assets: %s", ctx.headline()[:70])
    else:
        log.info(
            "stage 3 assets%s: %s",
            " (market-wide)" if ctx.market_wide else "",
            ", ".join(f"{link.asset}:{link.votes}v" for link in ctx.links),
        )
    return consensus


def aggregate_links(consensus: Consensus) -> list[AssetLink]:
    """Fold the panel's asset lists into one, ordered by strength of link."""
    answered = sum(1 for vote in consensus.votes if vote.ok)
    if not answered:
        return []

    # With a decimated panel, insisting on two mentions would discard everything.
    min_votes = ASSET_MIN_VOTES if answered >= 3 else 1

    buckets: dict[str, list[dict]] = {}
    for vote in consensus.votes:
        if not vote.ok:
            continue
        # One model naming the same asset twice votes once.
        seen: set[str] = set()
        for entry in vote.detail.get("assets", []):
            ticker = entry["asset"]
            if ticker in seen:
                continue
            seen.add(ticker)
            buckets.setdefault(ticker, []).append(entry)

    links: list[AssetLink] = []
    for ticker, entries in buckets.items():
        if len(entries) < min_votes:
            continue
        confidence = aggregate_numeric([entry["confidence"] for entry in entries]) or 0.5
        directs = sum(1 for entry in entries if entry["relation"] == Relation.DIRECT.value)
        links.append(
            AssetLink(
                asset=ticker,
                relation=Relation.DIRECT if directs * 2 >= len(entries) else Relation.INDIRECT,
                confidence=float(confidence),
                evidence=best_evidence([entry["evidence"] for entry in entries]),
                votes=len(entries),
            )
        )

    # Consensus first, then confidence: an asset four models named beats one that
    # two named with high conviction.
    links.sort(key=lambda link: (link.votes, link.confidence), reverse=True)
    return links[:MAX_ASSETS]


def _market_wide(consensus: Consensus) -> bool:
    votes = [bool(vote.detail.get("market_wide")) for vote in consensus.votes if vote.ok]
    return bool(votes) and sum(votes) * 2 > len(votes)


def _fallback_links(ctx: StageContext) -> list[AssetLink]:
    """Use the assets Stage 0 extracted when the panel named none.

    The article itself named a ticker, so there is something to analyse. The low
    confidence is honest — no model corroborated the link — and it flows through to
    the alert threshold.
    """
    assets = [a for a in (ctx.facts.assets or []) if normalize_ticker(a)]
    if not assets:
        return []
    log.info("stage 3 falling back to extracted assets: %s", ", ".join(assets[:3]))
    return [
        AssetLink(
            asset=normalize_ticker(asset),
            relation=Relation.DIRECT,
            confidence=0.4,
            evidence="named in the article; no panel corroboration",
            votes=0,
        )
        for asset in assets[:3]
    ]


__all__ = [
    "ASSET_MIN_VOTES",
    "MAX_ASSETS",
    "aggregate_links",
    "normalize_ticker",
    "parse",
    "run",
]
