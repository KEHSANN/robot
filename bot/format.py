"""Rendering a verdict as a Telegram message.

The format follows the spec's alert layout, and the ordering is not cosmetic: the
asset and direction come first because that is what a reader acts on, the expected
size and confidence next because they decide whether it is worth acting on, and the
mechanism after that because it is what makes the number checkable.

Two rules the rest of the module exists to keep:

**Every interpolated value is escaped.** Headlines contain ``&`` and ``<`` all the
time, and Telegram rejects the whole message on one unescaped character. An alert
lost to a stray ampersand is indistinguishable from an alert the system never
produced.

**Confidence and disagreement are always shown.** A verdict the panel split on must
not read like one it agreed on, so the model count and the audit trail are part of
the message rather than a debug detail.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from urllib.parse import urlsplit

from services.types import (
    AssetImpact,
    Causality,
    Direction,
    FactSet,
    Magnitude,
    NewsItem,
    Stage0Decision,
)

#: Telegram's hard limit. Messages are split below this, never truncated blindly.
MAX_MESSAGE = 4096

_MAGNITUDE_EMOJI = {
    Magnitude.LOW: "▫️",
    Magnitude.MEDIUM: "🔶",
    Magnitude.HIGH: "🔺",
    Magnitude.EXTREME: "🌋",
}

_URGENCY_EMOJI = {"critical": "🚨", "high": "⚡", "medium": "📊", "low": "📄"}

_CAUSALITY_LABEL = {
    Causality.DIRECT: "direct exposure",
    Causality.INDIRECT: "second-order",
    Causality.MACRO: "macro",
    Causality.LIQUIDITY: "liquidity",
    Causality.SENTIMENT: "sentiment",
    Causality.REGULATORY: "regulatory",
    Causality.FUNDAMENTAL: "fundamentals",
}

#: Depth of analysis, so a reader can tell a cheap verdict from an audited one.
_SOURCE_LABEL = {
    "stage4": "5-model panel",
    "stage5": "5-model panel + deep cross-check",
    "final": "deep cross-check + heavy panel",
}


def esc(value: object) -> str:
    """Escape for Telegram HTML. Applied to every value that reaches a message."""
    return html.escape(str(value if value is not None else ""), quote=False)


def _pct_range(impact: AssetImpact) -> str:
    """The expected move, signed by direction.

    Shown as a range rather than a point estimate because that is what the models
    were asked for, and collapsing it to a midpoint would imply precision the
    panel never claimed.
    """
    low = abs(impact.expected_low)
    high = abs(impact.expected_high)
    if high < low:
        low, high = high, low

    sign = {Direction.BULLISH: "+", Direction.BEARISH: "−"}.get(impact.direction, "±")
    if high - low < 0.05:
        return f"{sign}{high:.1f}%"
    return f"{sign}{low:.1f}% … {sign}{high:.1f}%"


def _event_line(facts: FactSet, news: NewsItem) -> str:
    """One line describing what happened, preferring extracted facts to the title."""
    if facts.headline:
        return facts.headline
    parts = [facts.entity, facts.action, facts.target]
    line = " ".join(part for part in parts if part).strip()
    return line if len(line) > 15 else news.title


def _sources_line(news: NewsItem, source_count: int) -> str:
    host = urlsplit(news.url).netloc.replace("www.", "") if news.url else ""
    name = news.source or host or "unknown"
    if source_count > 1:
        return f"{name} +{source_count - 1} more"
    return name


def _confidence_bar(confidence: float) -> str:
    filled = max(0, min(5, round(confidence * 5)))
    return "█" * filled + "░" * (5 - filled)


def _trail_line(trail: list[str]) -> str:
    """Which stages touched this verdict, for the footer."""
    return " → ".join(trail[-4:]) if trail else ""


def format_alert(alert, *, include_link: bool = True) -> str:
    """Render one alert. ``alert`` is a :class:`stages.pipeline.Alert`."""
    impact: AssetImpact = alert.impact
    facts: FactSet = alert.facts
    news: NewsItem = alert.news

    urgency = _URGENCY_EMOJI.get(alert.urgency, "📊")
    event_type = (facts.event_type or "OTHER").replace("_", " ")
    kind = "UPDATE" if alert.is_update else "NEW"

    lines = [
        f"{urgency} <b>{esc(event_type)}</b> · {esc(kind)}",
        f"<i>{esc(_event_line(facts, news))}</i>",
        "",
        f"{impact.direction.emoji} <b>{esc(impact.asset)}</b> — "
        f"<b>{esc(impact.direction.value)}</b>"
        + (f" ({esc(impact.relation.value.lower())})" if impact.relation.value != "DIRECT" else ""),
        f"{_MAGNITUDE_EMOJI.get(impact.magnitude, '▫️')} Expected move: "
        f"<b>{esc(_pct_range(impact))}</b> ({esc(impact.magnitude.value)})",
        f"🎯 Confidence: <b>{impact.confidence:.0%}</b> {_confidence_bar(impact.confidence)}",
        f"⏱ Horizon: <b>{esc(impact.horizon_label)}</b>",
    ]

    if impact.tradeable is False:
        lines.append("🚧 <b>Not tradeable</b> — move too small or panel unresolved")

    if impact.mechanism:
        lines += ["", f"💡 <b>Why:</b> {esc(impact.mechanism)}"]
    if impact.risks:
        lines.append(f"⚠️ <b>Risk:</b> {esc(impact.risks)}")
    if impact.key_uncertainty:
        lines.append(f"❓ <b>Unknown:</b> {esc(impact.key_uncertainty)}")
    if impact.notes:
        lines.append(f"🔀 <b>Revised:</b> {esc(impact.notes)}")

    footer = [
        "",
        f"🔗 {esc(_sources_line(news, alert.source_count))}",
        f"🧠 {esc(_SOURCE_LABEL.get(impact.source, impact.source))} · "
        f"{impact.model_count} models · {impact.agreement:.0%} agreement",
    ]
    if trail := _trail_line(alert.trail):
        footer.append(f"🧭 {esc(trail)}")
    footer.append(
        f"📈 {esc(_CAUSALITY_LABEL.get(impact.causality, impact.causality.value))}"
        f" · score {alert.score:.0f}"
    )
    if include_link and news.url:
        footer.append(f'<a href="{esc(news.url)}">read the source</a>')

    return "\n".join(lines + footer)


def format_plain(alert) -> str:
    """Parse-mode-free fallback.

    Telegram rejects a whole message when its markup is malformed. Rather than
    lose the alert, the sender retries with this — the same content, no tags.
    """
    impact: AssetImpact = alert.impact
    kind = "UPDATE" if alert.is_update else "NEW"
    lines = [
        f"[{(alert.facts.event_type or 'OTHER').replace('_', ' ')} · {kind}]",
        _event_line(alert.facts, alert.news),
        "",
        f"{impact.asset}: {impact.direction.value} {_pct_range(impact)} "
        f"({impact.magnitude.value})",
        f"confidence {impact.confidence:.0%} · horizon {impact.horizon_label} · "
        f"{impact.model_count} models",
    ]
    if impact.mechanism:
        lines += ["", f"Why: {impact.mechanism}"]
    if impact.risks:
        lines.append(f"Risk: {impact.risks}")
    if alert.news.url:
        lines += ["", alert.news.url]
    return "\n".join(lines)


def format_digest(alerts: list, *, limit: int = 8) -> str:
    """One line per alert, for a burst too large to send individually."""
    if not alerts:
        return "No alerts this pass."

    lines = [f"📋 <b>{len(alerts)} alerts</b>"]
    for alert in alerts[:limit]:
        impact = alert.impact
        lines.append(
            f"{impact.direction.emoji} <b>{esc(impact.asset)}</b> "
            f"{esc(_pct_range(impact))} · {impact.confidence:.0%} · "
            f"{esc(alert.news.title[:70])}"
        )
    if len(alerts) > limit:
        lines.append(f"…and {len(alerts) - limit} more")
    return "\n".join(lines)


def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Break a long message on paragraph, then line, then character boundaries.

    Splitting inside an HTML tag would make both halves unparseable, so the cuts
    are taken at blank lines and newlines first — where no tag can be open.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def format_status(stats: dict, counts: dict, keys: dict, *, started: datetime | None = None) -> str:
    """``/status``: what the pipeline has done and what is left standing."""
    lines = ["📊 <b>Pipeline status</b>"]

    if started is not None:
        uptime = datetime.now(timezone.utc) - started
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        lines.append(f"⏱ up {hours}h{remainder // 60:02d}m")

    funnel = [
        ("ingested", stats.get("ingested")),
        ("duplicates", stats.get("duplicates")),
        ("new events", stats.get("new_events")),
        ("updates", stats.get("updates")),
        ("analysed", stats.get("analysed")),
        ("stage 5", stats.get("escalated_stage5")),
        ("heavy panel", stats.get("escalated_final")),
        ("alerts", stats.get("alerts")),
    ]
    lines.append("")
    lines += [f"• {label}: <b>{value}</b>" for label, value in funnel if value is not None]

    if stats.get("llm_calls") is not None:
        failures = stats.get("llm_failures") or 0
        lines.append(f"• model calls: <b>{stats['llm_calls']}</b> ({failures} failed)")
    if stats.get("errors"):
        lines.append(f"• ⚠️ errors: <b>{stats['errors']}</b>")

    if counts:
        lines += ["", "🗄 <b>Stored</b>"]
        lines += [
            f"• {esc(name)}: {value}"
            for name, value in sorted(counts.items())
            if isinstance(value, int)
        ]

    if keys:
        lines += ["", "🔑 <b>Keys</b>"]
        for provider, pool in sorted(keys.items()):
            total = pool.get("total", 0)
            available = pool.get("available", 0)
            mark = "✅" if available else ("⚠️" if total else "⛔")
            lines.append(f"{mark} {esc(provider)}: {available}/{total} available")

    return "\n".join(lines)


def format_keys(snapshot: dict) -> str:
    """``/keys``: per-key health. Never contains a secret — only fingerprints."""
    if not snapshot:
        return "No key pools configured."

    lines = ["🔑 <b>API key health</b>"]
    for provider, pool in sorted(snapshot.items()):
        keys = pool.get("keys") or []
        if not keys:
            lines += ["", f"⛔ <b>{esc(provider)}</b> — no keys configured"]
            continue

        lines += ["", f"<b>{esc(provider)}</b> ({pool.get('available', 0)}/{pool.get('total', 0)} up)"]
        for key in keys:
            status = str(key.get("status", "?"))
            mark = {"healthy": "✅", "cooling": "🕒", "dead": "⛔"}.get(status, "❔")
            bits = [f"{mark} #{key.get('index')} <code>{esc(key.get('fingerprint'))}</code>"]
            if cooldown := key.get("cooldown_remaining"):
                bits.append(f"{cooldown:.0f}s")
            calls = key.get("success", 0) + key.get("failure", 0)
            if calls:
                bits.append(f"{key.get('success', 0)}/{calls} ok")
            if key.get("quota"):
                bits.append(f"quota×{key['quota']}")
            if key.get("rate_limited"):
                bits.append(f"429×{key['rate_limited']}")
            lines.append(" · ".join(bits))
            if status != "healthy" and key.get("last_error"):
                lines.append(f"   <i>{esc(str(key['last_error'])[:120])}</i>")

    return "\n".join(lines)


def format_recent(rows: list[dict], *, limit: int = 5) -> str:
    """``/last``: the newest verdicts, whether or not they were published.

    Rows come from ``Store.recent_analyses``, which joins the analysis to its
    event and article — so the description is the event headline, with the
    mechanism as a fallback for an analysis whose event row went away.
    """
    if not rows:
        return "Nothing analysed yet."

    lines = ["🗂 <b>Recent verdicts</b>"]
    for row in rows[:limit]:
        direction = Direction.parse(row.get("direction"))
        low = float(row.get("expected_low") or 0)
        high = float(row.get("expected_high") or 0)
        confidence = float(row.get("confidence") or 0)
        stage = row.get("deepest_stage") or 4
        when = row.get("created_at")
        stamp = when.strftime("%H:%M") if isinstance(when, datetime) else ""

        lines.append(
            f"{direction.emoji} <b>{esc(row.get('asset', '?'))}</b> "
            f"{abs(low):.1f}–{abs(high):.1f}% · {confidence:.0%} · s{stage}"
            + (f" · {esc(stamp)}" if stamp else "")
        )
        if headline := (row.get("headline") or row.get("mechanism")):
            lines.append(f"   <i>{esc(str(headline)[:90])}</i>")
    return "\n".join(lines)


def format_decision(decision: Stage0Decision, reason: str) -> str:
    """Used by ``/check``: what Stage 0 concluded about a pasted headline."""
    mark = {
        Stage0Decision.NEW: "🆕",
        Stage0Decision.UPDATE: "🔁",
        Stage0Decision.DUPLICATE: "♻️",
    }.get(decision, "❔")
    return f"{mark} <b>{esc(decision.value)}</b>\n{esc(reason)}"


__all__ = [
    "MAX_MESSAGE",
    "esc",
    "format_alert",
    "format_decision",
    "format_digest",
    "format_keys",
    "format_plain",
    "format_recent",
    "format_status",
    "split_message",
]
