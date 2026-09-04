"""Fact extraction for Stage 0.

One model, not a panel: "what does this article say" is transcription, and a
five-model vote would spend five times the tokens to agree with itself. The
:data:`services.models.EXTRACTORS` list is an ordered fallback chain that crosses
providers, so a total Gemini outage degrades to Groq rather than stalling.

When every extractor fails there is still a usable answer — :func:`heuristic_facts`
reads what it can from the headline with regexes. The identity it produces is
coarser, so it will under-merge (two reports of one event may both look NEW), but
under-merging costs a duplicate analysis while returning nothing costs the alert.
"""

from __future__ import annotations

import logging
import re

from services.jsonparse import as_float
from services.llm import LLMClient, LLMResult
from services.models import EXTRACTORS, ModelSpec
from services.prompts import FACTS_SYSTEM, facts_user
from services.types import FactSet, NewsItem

log = logging.getLogger(__name__)

#: Tickers we recognise without help from a model. Deliberately short: this is
#: the fallback path, and a wrong ticker is worse than no ticker.
KNOWN_ASSETS: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc", "xbt"),
    "ETH": ("ethereum", "eth", "ether"),
    "SOL": ("solana", "sol"),
    "XRP": ("xrp", "ripple"),
    "BNB": ("bnb", "binance coin"),
    "ADA": ("cardano", "ada"),
    "DOGE": ("dogecoin", "doge"),
    "AVAX": ("avalanche", "avax"),
    "LINK": ("chainlink", "link"),
    "DOT": ("polkadot", "dot"),
    "MATIC": ("polygon", "matic"),
    "TRX": ("tron", "trx"),
    "LTC": ("litecoin", "ltc"),
    "SHIB": ("shiba inu", "shib"),
    "TON": ("toncoin", "ton"),
    "SUI": ("sui",),
    "APT": ("aptos", "apt"),
    "ARB": ("arbitrum", "arb"),
    "OP": ("optimism",),
    "USDT": ("tether", "usdt"),
    "USDC": ("usdc", "circle"),
}

#: Headline patterns -> event type, for the heuristic fallback.
_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\betf\b.*\b(inflow|outflow|flows?)\b|\b(inflow|outflow)s?\b.*\betf\b", "ETF_FLOWS"),
    (r"\betf\b.*\b(approv|reject|deni|deadline|filing|list)", "ETF_APPROVAL"),
    (r"\b(hack|exploit|breach|drain|stolen|steal)", "EXCHANGE_HACK"),
    (r"\b(sue|sued|lawsuit|court|judge|settle|settlement)", "LAWSUIT"),
    (r"\b(sec|cftc|regulator|regulation|regulatory|mica|ban|legislat|bill)\b", "REGULATORY"),
    (r"\b(charge|indict|enforce|fine|penalt|seiz|sanction)", "ENFORCEMENT"),
    (r"\b(bankrupt|insolven|chapter 11|liquidat.*(firm|company))", "BANKRUPTCY"),
    (r"\bliquidat(ed|ion)s?\b", "LIQUIDATION"),
    (r"\b(list|listing)\b.*\b(exchange|coinbase|binance|upbit)", "LISTING"),
    (r"\bdelist", "DELISTING"),
    (r"\b(partner|collaborat|integrat)", "PARTNERSHIP"),
    (r"\b(raise[sd]?|funding|series [a-d]|seed round|valuation)\b", "FUNDING"),
    (r"\b(acquir|merger|buyout|takeover)", "ACQUISITION"),
    (r"\b(upgrade|hard fork|mainnet|testnet|halving)\b", "NETWORK_UPGRADE"),
    (r"\b(outage|downtime|halted|suspend)", "OUTAGE"),
    (r"\b(whale|moved|transferred)\b.*\b(btc|eth|million|billion)", "WHALE_MOVEMENT"),
    (r"\b(treasury|buys?|bought|purchase[sd]?|adds?)\b.*\b(bitcoin|btc|eth)\b", "TREASURY_PURCHASE"),
    (r"\b(fed|fomc|interest rate|cpi|inflation|jobs report|tariff)", "MACRO"),
    (r"\b(adopt|legal tender|accept.*payment)", "ADOPTION"),
    (r"\b(ceo|cfo|resign|step down|appoint|hire)", "PERSONNEL"),
    (r"\b(surge|plunge|rally|crash|drop|soar|dip|price)\b", "PRICE_ACTION"),
)

_AMOUNT_RE = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand|[tbmk])?\b",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"([\d]+(?:\.\d+)?)\s?%")
_MULTIPLIERS = {
    "trillion": 1e12, "t": 1e12,
    "billion": 1e9, "b": 1e9,
    "million": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}


class FactExtractor:
    """Extracts a :class:`FactSet` from an article, with fallbacks all the way
    down to regexes."""

    def __init__(self, client: LLMClient, extractors: tuple[ModelSpec, ...] | None = None) -> None:
        self.client = client
        self.extractors = extractors or EXTRACTORS
        self.calls = 0
        self.llm_failures = 0
        self.heuristic_fallbacks = 0

    async def extract(self, news: NewsItem) -> tuple[FactSet, LLMResult | None]:
        """Return the facts plus the result of the call that produced them.

        The result is returned so the caller can log which model answered and how
        long it took; ``None`` means the heuristic path was used.
        """
        user = facts_user(news)
        last: LLMResult | None = None

        for spec in self.extractors:
            self.calls += 1
            result = await self.client.complete_json(spec, FACTS_SYSTEM, user)
            last = result
            if not result.ok or result.data is None:
                self.llm_failures += 1
                log.debug("fact extraction via %s failed: %s", spec.label, result.error)
                continue

            facts = FactSet.from_json(result.data)
            facts = self._repair(facts, news)
            if facts.action or facts.entity or facts.assets:
                return facts, result

            log.debug("%s returned an empty fact set; trying next extractor", spec.label)

        self.heuristic_fallbacks += 1
        log.warning("falling back to heuristic facts for %r", news.title[:80])
        return heuristic_facts(news), last

    def _repair(self, facts: FactSet, news: NewsItem) -> FactSet:
        """Fill in what the model left blank but the text plainly states.

        Models routinely omit the asset when the headline names it, and just as
        routinely return "$742 million" as the string it was written as. Both are
        cheap to fix here and expensive to leave broken, because both feed the
        identity hash.
        """
        text = news.text

        if not facts.assets:
            facts.assets = detect_assets(text)

        # A model asked for plain numbers still sometimes answers "742M".
        if facts.amount is None:
            facts.amount = detect_amount(news.title) or detect_amount(text[:1200])
        if facts.percentage is None:
            facts.percentage = detect_percentage(news.title)

        if not facts.event_type or facts.event_type == "OTHER":
            facts.event_type = detect_event_type(text)
        if not facts.headline:
            facts.headline = news.title.strip()[:200]
        if not facts.status:
            facts.status = "REPORTED"

        facts.assets = _dedupe_upper(facts.assets)
        return facts


# --------------------------------------------------------------------------- #
# heuristics — also used to repair partial LLM output
# --------------------------------------------------------------------------- #

def detect_assets(text: str) -> list[str]:
    """Tickers named in the text, ordered by first mention."""
    lowered = (text or "").lower()
    found: list[tuple[int, str]] = []
    for ticker, aliases in KNOWN_ASSETS.items():
        best: int | None = None
        for alias in aliases:
            match = re.search(rf"\b{re.escape(alias)}\b", lowered)
            if match and (best is None or match.start() < best):
                best = match.start()
        if best is not None:
            found.append((best, ticker))
    found.sort()
    return [ticker for _, ticker in found[:6]]


def detect_amount(text: str) -> float | None:
    """First dollar figure in the text, expanded to plain USD."""
    match = _AMOUNT_RE.search(text or "")
    if not match:
        return None
    value = as_float(match.group(1).replace(",", ""))
    if value is None:
        return None
    suffix = (match.group(2) or "").lower()
    return value * _MULTIPLIERS.get(suffix, 1.0)


def detect_percentage(text: str) -> float | None:
    match = _PCT_RE.search(text or "")
    return as_float(match.group(1)) if match else None


def detect_event_type(text: str) -> str:
    lowered = (text or "").lower()
    for pattern, event_type in _TYPE_PATTERNS:
        if re.search(pattern, lowered):
            return event_type
    return "OTHER"


def heuristic_facts(news: NewsItem) -> FactSet:
    """Last-resort facts built from the headline alone.

    Identity is coarse here — entity is whatever proper noun leads the headline —
    so this path under-merges rather than over-merges. That is the safe direction:
    a missed merge costs one duplicate analysis, a wrong merge silently buries a
    real event inside an unrelated one.
    """
    title = news.title.strip()
    text = news.text

    return FactSet(
        event_type=detect_event_type(text),
        entity=_leading_proper_noun(title),
        assets=detect_assets(text),
        action=_leading_verb(title),
        status="REPORTED",
        amount=detect_amount(title) or detect_amount(text[:1200]),
        percentage=detect_percentage(title),
        key_claims=[title[:200]] if title else [],
        headline=title[:200],
    )


def _leading_proper_noun(title: str) -> str:
    """The first capitalised run in the headline, which is usually the actor."""
    match = re.search(r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,2})", title or "")
    if not match:
        return ""
    candidate = match.group(1).strip()
    # A headline that opens with a stopword ("The SEC has…") yields junk here.
    if candidate.lower() in ("the", "a", "an", "this", "these", "new", "why", "how"):
        rest = (title or "")[match.end():]
        return _leading_proper_noun(rest)
    return candidate[:80]


_VERB_HINT = re.compile(
    r"\b(approv\w*|reject\w*|deni\w*|sue[sd]?|sued|charg\w*|fil\w*|launch\w*|list\w*|"
    r"delist\w*|hack\w*|exploit\w*|buy[s]?|bought|sell[s]?|sold|rais\w*|acquir\w*|"
    r"partner\w*|halt\w*|suspend\w*|resign\w*|announc\w*|report\w*|surge[sd]?|"
    r"plunge[sd]?|jump\w*|drop\w*|crash\w*|rall\w*|add[s]?|added|move[sd]?|"
    r"transfer\w*|upgrad\w*|ban[s|ned]*|fine[sd]?|settl\w*|warn\w*|propos\w*)\b",
    re.IGNORECASE,
)


def _leading_verb(title: str) -> str:
    match = _VERB_HINT.search(title or "")
    return match.group(1).upper() if match else ""


def _dedupe_upper(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in seen:
            seen.append(ticker)
    return seen[:6]


__all__ = [
    "FactExtractor",
    "KNOWN_ASSETS",
    "detect_amount",
    "detect_assets",
    "detect_event_type",
    "detect_percentage",
    "heuristic_facts",
]
