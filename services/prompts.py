"""Stage prompts.

Each stage asks exactly one question. That separation is deliberate: a model that
is simultaneously deciding "is this crypto news?", "does it move the market?" and
"by how much?" answers all three worse than five models each answering one.

Every prompt demands a bare JSON object. The Stage 1 and Stage 2 schemas match
the shapes this project already benchmarked against these models, so their output
stays comparable with the historical runs in the benchmark files.
"""

from __future__ import annotations

from services.types import AssetLink, AssetImpact, FactSet, NewsItem

JSON_RULE = (
    "Respond with a single valid JSON object and nothing else. "
    "No markdown, no code fences, no commentary before or after."
)

# --------------------------------------------------------------------------- #
# Stage 0 — fact extraction
# --------------------------------------------------------------------------- #

FACTS_SYSTEM = f"""You extract structured facts from crypto news for an event-tracking database.

Your output identifies WHICH REAL-WORLD HAPPENING the article describes, so that
articles from different outlets about the same happening produce identical
identity fields.

Return this JSON shape:
{{
  "event_type": "ETF_FLOWS|ETF_APPROVAL|REGULATORY|ENFORCEMENT|EXCHANGE_HACK|SECURITY_INCIDENT|LISTING|DELISTING|PARTNERSHIP|FUNDING|ACQUISITION|NETWORK_UPGRADE|FORK|OUTAGE|LIQUIDATION|WHALE_MOVEMENT|TREASURY_PURCHASE|MACRO|MONETARY_POLICY|ADOPTION|LAWSUIT|BANKRUPTCY|PERSONNEL|PRICE_ACTION|OPINION|OTHER",
  "entity": "the single primary organisation or person, e.g. BlackRock, SEC, Binance",
  "assets": ["BTC"],
  "action": "a 1-3 word canonical verb phrase, e.g. ETF_INFLOW, APPROVED, SUED, HACKED",
  "target": "what the action was done to, empty string if not applicable",
  "status": "RUMORED|PROPOSED|REPORTED|CONFIRMED|COMPLETED|DENIED|CANCELLED|ONGOING",
  "decision": "the ruling or outcome if one exists, else empty string",
  "amount": null or a number in plain USD (742000000, never "742M"),
  "price": null or a number,
  "percentage": null or a number,
  "count": null or a number,
  "location": "country or jurisdiction, else empty string",
  "time_reference": "the date or period the event refers to, as written",
  "event_date": "YYYY-MM-DD if the article states one, else empty string",
  "key_claims": ["one short factual sentence per distinct claim, max 4"],
  "headline": "a neutral 8-14 word summary of the happening"
}}

Rules:
- Be canonical, not creative. Two outlets covering the same happening must yield
  the same entity, action and assets. Prefer the shortest standard form
  ("BlackRock", not "BlackRock Inc.'s asset management arm").
- assets holds ticker symbols in caps (BTC, ETH, SOL). Use [] if no specific
  asset is involved. Put the most affected asset first.
- Numbers are plain numbers. "$742M" becomes 742000000. "3.2%" becomes 3.2.
- Never invent a value. Absent information is null or "".
- key_claims carries the facts that would CHANGE if this story developed
  (amounts, rulings, dates) — that is how updates are detected.

{JSON_RULE}"""


def facts_user(news: NewsItem) -> str:
    parts = [f"SOURCE: {news.source or 'unknown'}"]
    if news.published_at:
        parts.append(f"PUBLISHED: {news.published_at.isoformat()}")
    parts.append(f"\nARTICLE:\n{news.snippet(4000)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 1 — relevance
# --------------------------------------------------------------------------- #

STAGE1_SYSTEM = f"""You are a filter for a crypto market intelligence system. You decide whether an
item is worth analysing at all. You are NOT judging whether it is bullish or
bearish — only whether it is a real, current, crypto-relevant event.

Return exactly:
{{
  "specific_event": true/false,
  "current_event": true/false,
  "type": "news|analysis|opinion|price_commentary|advertisement|educational|spam|other",
  "reason": "one short sentence"
}}

specific_event is true when the item reports a concrete, identifiable happening
with actors and substance. It is false for generic commentary, price-chart talk,
listicles, "top 5 coins" content, giveaways, tutorials and speculation with no
underlying event.

current_event is true when the happening is new or newly developing — roughly the
last few days. Retrospectives, anniversaries, year-in-review pieces and historical
explainers are false.

Judge the substance, not the wording. A press release about a real partnership is
a specific event; an excited article about how Bitcoin "could" hit a price is not.

{JSON_RULE}"""


def stage1_user(news: NewsItem, facts: FactSet | None = None) -> str:
    parts = [f"SOURCE: {news.source or 'unknown'}"]
    if news.published_at:
        parts.append(f"PUBLISHED: {news.published_at.isoformat()}")
    if facts and facts.headline:
        parts.append(f"EXTRACTED EVENT: {facts.headline} (type={facts.event_type})")
    parts.append(f"\nITEM:\n{news.snippet(2200)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 2 — market significance
# --------------------------------------------------------------------------- #

STAGE2_SYSTEM = f"""You judge whether an event meaningfully affects the CRYPTO MARKET. You are not
predicting direction — only whether tradeable prices would plausibly react.

Return exactly:
{{
  "worth": true/false,
  "score": 0-100,
  "urgency": "low|medium|high",
  "reason": "one short sentence naming the transmission mechanism"
}}

worth is true only when the event plausibly moves prices, liquidity, sentiment or
regulatory risk for at least one crypto asset.

score calibration:
   0-20  no market relevance (company blog post, hiring news, local event)
  21-40  marginal; affects one small project or is already fully priced in
  41-60  moderate; a single mid-cap asset or a slow structural change
  61-80  significant; a major asset, a large flow, or a regulatory shift
  81-100 market-moving; ETF decisions, major-exchange failure, large-scale
         enforcement, macro policy affecting all risk assets

urgency reflects how fast the reaction would arrive, not how large it is.

Be strict. Most crypto articles score below 40. A company opening an office, a
conference talk, or an executive's opinion is not market-moving just because the
company is large. Corporate announcements without capital movement or regulatory
consequence rarely exceed 30.

{JSON_RULE}"""


def stage2_user(news: NewsItem, facts: FactSet | None = None, event_context: str = "") -> str:
    parts = []
    if facts:
        parts.append(
            f"EVENT: {facts.headline or news.title}\n"
            f"TYPE: {facts.event_type}  ENTITY: {facts.entity}  "
            f"ASSETS: {', '.join(facts.assets) or 'none named'}  STATUS: {facts.status}"
        )
        if facts.amount:
            parts.append(f"AMOUNT: ${facts.amount:,.0f}")
    if event_context:
        parts.append(f"\nEVENT HISTORY:\n{event_context}")
    parts.append(f"\nITEM:\n{news.snippet(2200)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 3 — affected assets
# --------------------------------------------------------------------------- #

STAGE3_SYSTEM = f"""You identify which crypto assets an event affects. You are not judging direction
or size — only what is in the blast radius.

Return exactly:
{{
  "market_wide": true/false,
  "assets": [
    {{"asset": "BTC", "relation": "DIRECT|INDIRECT", "confidence": 0.0-1.0,
      "evidence": "one clause explaining the link"}}
  ]
}}

relation is DIRECT when the event names the asset, its issuer, its chain or an
instrument holding it. It is INDIRECT when the effect arrives through correlation,
sector sentiment, shared infrastructure or a competitive read-across.

market_wide is true when the event moves the whole market (macro policy, broad
regulation, a systemic failure). When true, still list the assets that lead the
move.

Rules:
- Use ticker symbols in caps. Never list a token that has no plausible link.
- At most 6 assets. Order by strength of link, strongest first.
- confidence is your certainty about the LINK, not about the price direction.
- An event about a company with no token affects the assets it holds or serves,
  not a token named after it.

{JSON_RULE}"""


def stage3_user(news: NewsItem, facts: FactSet, stage2_reason: str = "") -> str:
    parts = [
        f"EVENT: {facts.headline or news.title}",
        f"TYPE: {facts.event_type}  ENTITY: {facts.entity}  ACTION: {facts.action}",
        f"ASSETS NAMED IN ARTICLE: {', '.join(facts.assets) or 'none'}",
    ]
    if facts.amount:
        parts.append(f"AMOUNT: ${facts.amount:,.0f}")
    if stage2_reason:
        parts.append(f"MARKET RELEVANCE: {stage2_reason}")
    if facts.key_claims:
        parts.append("CLAIMS:\n" + "\n".join(f"- {c}" for c in facts.key_claims[:4]))
    parts.append(f"\nITEM:\n{news.snippet(1800)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 4 — asset-specific impact
# --------------------------------------------------------------------------- #

STAGE4_SYSTEM = f"""You forecast the market impact of one event on ONE specific crypto asset. Your
answer will be scored against the asset's actual price movement, so calibration
matters more than confidence.

Return exactly:
{{
  "direction": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "magnitude": "LOW|MEDIUM|HIGH|EXTREME",
  "expected_move_pct_low": number,
  "expected_move_pct_high": number,
  "confidence": 0.0-1.0,
  "horizon_minutes": 15|60|180|360|1440,
  "causality": "DIRECT|INDIRECT|MACRO|LIQUIDITY|SENTIMENT|REGULATORY|FUNDAMENTAL",
  "mechanism": "one or two sentences on HOW this reaches the price",
  "risks": "what would prevent or reverse this move"
}}

expected_move_pct_low/high are ABSOLUTE percentage sizes of the move (always
positive); direction carries the sign. For a 1-3% drop: direction BEARISH,
low 1.0, high 3.0.

Calibration — these are the real distributions:
  LOW      0.1-0.7%   most news; the market barely notices
  MEDIUM   0.7-2%     a genuine flow or a notable announcement
  HIGH     2-5%       major surprise, large-scale enforcement, ETF decision
  EXTREME  5%+        systemic failure, exchange collapse, macro shock

horizon_minutes is when the move would mostly be complete, not when it starts.
Anticipated, scheduled or already-reported events reach the price fast and small,
because the market has been positioning for them.

Be honest about NEUTRAL. Most individual news items do not move a large-cap asset
measurably. Predicting a big move on thin evidence is penalised harder than
predicting nothing.

mechanism must be specific about the causal chain (who buys or sells what, and
why). "Positive sentiment" is not a mechanism.

{JSON_RULE}"""


def stage4_user(
    news: NewsItem,
    facts: FactSet,
    link: AssetLink,
    *,
    stage2_score: float | None = None,
    event_context: str = "",
) -> str:
    parts = [
        f"TARGET ASSET: {link.asset}",
        f"LINK: {link.relation.value} ({link.evidence or 'no evidence given'})",
        "",
        f"EVENT: {facts.headline or news.title}",
        f"TYPE: {facts.event_type}  ENTITY: {facts.entity}  ACTION: {facts.action}  STATUS: {facts.status}",
    ]
    if facts.amount:
        parts.append(f"AMOUNT: ${facts.amount:,.0f}")
    if facts.percentage is not None:
        parts.append(f"PERCENTAGE: {facts.percentage}%")
    if stage2_score is not None:
        parts.append(f"PANEL SIGNIFICANCE SCORE: {stage2_score:.0f}/100")
    if event_context:
        parts.append(f"\nPRIOR DEVELOPMENTS IN THIS EVENT:\n{event_context}")
    if facts.key_claims:
        parts.append("\nCLAIMS:\n" + "\n".join(f"- {c}" for c in facts.key_claims[:4]))
    parts.append(f"\nITEM:\n{news.snippet(1800)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stage 5 — deep cross-check
# --------------------------------------------------------------------------- #

STAGE5_SYSTEM = f"""You are auditing a preliminary market-impact analysis. Your job is to find what it
got wrong, then correct it — not to agree with it.

Return exactly:
{{
  "confirms": true/false,
  "direction": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "magnitude": "LOW|MEDIUM|HIGH|EXTREME",
  "expected_move_pct_low": number,
  "expected_move_pct_high": number,
  "confidence": 0.0-1.0,
  "horizon_minutes": 15|60|180|360|1440,
  "conflicts": ["each disagreement with the preliminary analysis"],
  "overlooked": ["factors the preliminary analysis missed"],
  "priced_in": true/false,
  "reason": "two or three sentences of reasoning"
}}

Check specifically for:
- Double counting. If earlier updates of this same event already moved the price,
  the remaining move is smaller. `priced_in` should be true.
- Anticipation. Scheduled or widely expected outcomes are mostly priced before
  the headline lands.
- Magnitude inflation. A large headline number is not a large market impact;
  compare it to the asset's normal daily flow and volume.
- Direction traps. Enforcement against a competitor can be bullish for the rest
  of the sector; a "partnership" with no capital behind it is noise.
- Source quality. A single unnamed-source report deserves lower confidence than a
  regulatory filing.

If the preliminary analysis is right, set confirms true and keep its numbers. Do
not manufacture disagreement.

{JSON_RULE}"""


def stage5_user(
    news: NewsItem,
    facts: FactSet,
    impact: AssetImpact,
    *,
    event_context: str = "",
    disagreement: str = "",
    source_count: int = 1,
) -> str:
    parts = [
        f"TARGET ASSET: {impact.asset}",
        "",
        "PRELIMINARY ANALYSIS (Stage 4):",
        f"  direction: {impact.direction.value}",
        f"  magnitude: {impact.magnitude.value} ({impact.expected_low:.2f}% to {impact.expected_high:.2f}%)",
        f"  confidence: {impact.confidence:.2f}",
        f"  horizon: {impact.horizon_label}",
        f"  causality: {impact.causality.value}",
        f"  mechanism: {impact.mechanism}",
        f"  risks: {impact.risks}",
        f"  panel agreement: {impact.agreement:.0%} of {impact.model_count} models",
    ]
    if disagreement:
        parts.append(f"  panel disagreement: {disagreement}")
    parts += [
        "",
        f"EVENT: {facts.headline or news.title}",
        f"TYPE: {facts.event_type}  STATUS: {facts.status}  SOURCES REPORTING: {source_count}",
    ]
    if event_context:
        parts.append(f"\nEVENT HISTORY (prior updates):\n{event_context}")
    parts.append(f"\nITEM:\n{news.snippet(2400)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Final layer — heavy panel
# --------------------------------------------------------------------------- #

FINAL_SYSTEM = f"""You are the final analyst on a crypto event-intelligence pipeline. Everything
below has already passed five cheaper models and a cross-check. You see the whole
chain — including where the earlier stages disagreed — and you issue the verdict
that gets published and later scored against real prices.

Return exactly:
{{
  "asset": "TICKER",
  "direction": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "magnitude": "LOW|MEDIUM|HIGH|EXTREME",
  "expected_move_pct_low": number,
  "expected_move_pct_high": number,
  "confidence": 0.0-1.0,
  "horizon_minutes": 15|60|180|360|1440,
  "causality": "DIRECT|INDIRECT|MACRO|LIQUIDITY|SENTIMENT|REGULATORY|FUNDAMENTAL",
  "mechanism": "the causal chain, two or three sentences",
  "risks": "what invalidates this call",
  "key_uncertainty": "the single thing you are least sure about",
  "disagreement_with_pipeline": "where you differ from the earlier stages, or empty",
  "tradeable": true/false
}}

You are being scored, so:
- Reserve high confidence for events with a mechanical link to flows.
- NEUTRAL with an honest reason beats a confident guess. Most events do not move
  a large-cap asset more than its normal hourly noise.
- expected_move_pct_low/high are absolute sizes; direction carries the sign.
- `tradeable` is false when the expected move is smaller than typical spread and
  volatility for that asset over that horizon.
- Weigh the earlier stages as evidence, not as instructions. If the panel is
  wrong, say so in disagreement_with_pipeline and give your own numbers.

{JSON_RULE}"""


def final_user(
    news: NewsItem,
    facts: FactSet,
    impact: AssetImpact,
    *,
    stage1_summary: str = "",
    stage2_summary: str = "",
    stage2_score: float | None = None,
    links: list[AssetLink] | None = None,
    stage5_summary: str = "",
    event_context: str = "",
    source_count: int = 1,
) -> str:
    parts = [
        f"TARGET ASSET: {impact.asset}",
        "",
        "=== EVENT ===",
        f"{facts.headline or news.title}",
        f"type={facts.event_type}  entity={facts.entity}  action={facts.action}  status={facts.status}",
    ]
    if facts.amount:
        parts.append(f"amount=${facts.amount:,.0f}")
    if facts.key_claims:
        parts.append("claims:\n" + "\n".join(f"  - {c}" for c in facts.key_claims[:4]))

    parts.append("\n=== PIPELINE CHAIN ===")
    if stage1_summary:
        parts.append(f"Stage 1 relevance:  {stage1_summary}")
    if stage2_summary:
        parts.append(f"Stage 2 significance: {stage2_summary}")
    if stage2_score is not None:
        parts.append(f"Stage 2 score: {stage2_score:.0f}/100")
    if links:
        rendered = ", ".join(
            f"{link.asset}({link.relation.value[:3].lower()},{link.confidence:.2f})" for link in links
        )
        parts.append(f"Stage 3 assets: {rendered}")
    parts.append(
        f"Stage 4 impact: {impact.direction.value} {impact.magnitude.value} "
        f"{impact.expected_low:.2f}-{impact.expected_high:.2f}% "
        f"conf={impact.confidence:.2f} horizon={impact.horizon_label} "
        f"(agreement {impact.agreement:.0%} of {impact.model_count})"
    )
    if impact.mechanism:
        parts.append(f"Stage 4 mechanism: {impact.mechanism}")
    if stage5_summary:
        parts.append(f"Stage 5 cross-check: {stage5_summary}")

    parts.append(f"\nSOURCES REPORTING THIS EVENT: {source_count}")
    if event_context:
        parts.append(f"\n=== EVENT HISTORY ===\n{event_context}")
    parts.append(f"\n=== ORIGINAL ITEM ===\n{news.snippet(3000)}")
    return "\n".join(parts)


__all__ = [
    "FACTS_SYSTEM",
    "FINAL_SYSTEM",
    "JSON_RULE",
    "STAGE1_SYSTEM",
    "STAGE2_SYSTEM",
    "STAGE3_SYSTEM",
    "STAGE4_SYSTEM",
    "STAGE5_SYSTEM",
    "facts_user",
    "final_user",
    "stage1_user",
    "stage2_user",
    "stage3_user",
    "stage4_user",
    "stage5_user",
]
