"""Deciding which verdicts get sent, and recording what happened.

The pipeline produces alerts; this module is the gate between them and a reader's
phone. It exists because the two failure modes here are asymmetric and both bad:
sending too much trains the reader to ignore the bot, and silently dropping an
alert leaves nothing to explain the gap.

So three filters, in cost order:

**Score threshold.** ``TELEGRAM_MIN_ALERT_SCORE`` is the line between "the panel
had an opinion" and "the panel had an opinion worth interrupting someone for".

**Repeat suppression.** An UPDATE means the event's state changed, which is
usually worth sending — but not when the *verdict* is unchanged. A story that
updates six times while the analysis keeps saying BTC / BULLISH / MEDIUM should
produce one message, not six. The in-process check compares the verdict itself;
the database check is the backstop that survives a restart.

**Escalation override.** A verdict that has since been through the deep
cross-check or the heavy panel is allowed to re-alert even for an already-sent
asset, because "we looked harder and it is worse than we said" is new
information.

Every attempt is written to ``telegram_alerts``, including failures. An alert that
was formatted but never delivered is exactly the thing that would otherwise be
invisible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bot.format import format_alert, format_digest, format_plain
from bot.telegram import SendResult, TelegramClient
from database.base import Store
from services.config import Settings, settings as global_settings
from services.types import Magnitude

log = logging.getLogger(__name__)

#: Verdicts whose depth justifies a second message about the same asset.
_DEEP_SOURCES = ("stage5", "final")

#: Above this many alerts in one pass, send a digest instead of individual
#: messages. A 20-message burst is unreadable and gets the bot rate limited.
DIGEST_THRESHOLD = 6


@dataclass
class PublishReport:
    """What one publish pass did, for the run log and ``/status``."""

    considered: int = 0
    below_threshold: int = 0
    suppressed: int = 0
    sent: int = 0
    failed: int = 0
    digested: bool = False
    results: list[SendResult] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.sent} sent"]
        if self.digested:
            parts.append("as digest")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.below_threshold:
            parts.append(f"{self.below_threshold} below threshold")
        if self.suppressed:
            parts.append(f"{self.suppressed} repeats")
        return ", ".join(parts) + f" (of {self.considered})"


class Publisher:
    """Publishes alerts to the configured chats and records the outcome."""

    def __init__(
        self,
        client: TelegramClient,
        store: Store | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or global_settings
        self.client = client
        self.store = store
        #: (event_id, asset) -> the verdict last published, for repeat detection
        self._published: dict[tuple[int | str, str], tuple[str, str, int]] = {}

    # ------------------------------------------------------------------ gating

    def _verdict_key(self, alert) -> tuple[int | str, str]:
        """Identity of the thing being alerted on.

        Falls back to the news id when there is no event: without an identity,
        every alert is its own, which errs toward sending rather than silence.
        """
        event = alert.event_id if alert.event_id is not None else f"news:{alert.news_id}"
        return (event, alert.impact.asset.upper())

    def _verdict_shape(self, alert) -> tuple[str, str, int]:
        """The part of a verdict that has to change for it to be worth resending.

        Confidence is bucketed to 10% because a panel re-running on new wording
        will move it a point or two without meaning anything different.
        """
        impact = alert.impact
        return (
            impact.direction.value,
            impact.magnitude.value,
            int(round(impact.confidence * 10)),
        )

    async def _should_send(self, alert, report: PublishReport) -> bool:
        threshold = self.settings.telegram.min_alert_score
        if alert.score < threshold:
            report.below_threshold += 1
            log.debug("alert for %s scored %.0f < %d, holding",
                      alert.impact.asset, alert.score, threshold)
            return False

        key = self._verdict_key(alert)
        shape = self._verdict_shape(alert)
        escalated = alert.impact.source in _DEEP_SOURCES

        previous = self._published.get(key)
        if previous is not None and previous == shape and not escalated:
            report.suppressed += 1
            log.debug("alert for %s repeats an unchanged verdict, holding",
                      alert.impact.asset)
            return False

        # Cross-restart backstop. Only consulted when the process has no memory
        # of this pair, so the common case costs no query.
        if previous is None and self.store is not None and alert.event_id is not None:
            try:
                already = await self.store.already_alerted(
                    alert.event_id, alert.impact.asset
                )
            except Exception:
                log.exception("already_alerted check failed; sending anyway")
                already = False

            if already and not escalated:
                report.suppressed += 1
                log.debug("alert for %s already sent before restart, holding",
                          alert.impact.asset)
                return False

        return True

    # ----------------------------------------------------------------- sending

    async def publish(self, alerts: list) -> PublishReport:
        """Filter, send and record a batch of alerts."""
        report = PublishReport(considered=len(alerts))

        if not alerts:
            return report
        if not self.settings.telegram.enabled:
            log.info("telegram disabled; %d alerts not sent", len(alerts))
            return report
        if not self.settings.telegram.configured:
            log.warning(
                "telegram not configured (need TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_IDS); %d alerts not sent", len(alerts)
            )
            return report

        eligible = [alert for alert in alerts if await self._should_send(alert, report)]
        if not eligible:
            log.info("publish: %s", report.summary())
            return report

        if len(eligible) > DIGEST_THRESHOLD:
            await self._publish_digest(eligible, report)
        else:
            for alert in eligible:
                await self._publish_one(alert, report)

        log.info("publish: %s", report.summary())
        return report

    async def _publish_one(self, alert, report: PublishReport) -> None:
        text = format_alert(alert)
        plain = format_plain(alert)
        delivered = False

        for chat_id in self.settings.telegram.chat_ids:
            result = await self.client.send(
                chat_id, text,
                plain_fallback=plain,
                silent=alert.urgency == "low",
            )
            report.results.append(result)

            if result.sent:
                report.sent += 1
                delivered = True
            else:
                report.failed += 1

            await self._record(alert, chat_id, text, result)

        if delivered:
            # Only remember a verdict that actually reached someone, so a failed
            # send is retried next pass instead of being suppressed as a repeat.
            self._published[self._verdict_key(alert)] = self._verdict_shape(alert)

    async def _publish_digest(self, alerts: list, report: PublishReport) -> None:
        """One message for a burst.

        The individual verdicts are still written to the database, so nothing is
        lost — only the notification is collapsed.
        """
        report.digested = True
        text = format_digest(alerts)

        for chat_id in self.settings.telegram.chat_ids:
            result = await self.client.send(chat_id, text)
            report.results.append(result)
            if result.sent:
                report.sent += 1
            else:
                report.failed += 1

            if self.store is not None:
                await self._record_raw(
                    chat_id, text, result, kind="digest",
                    asset=f"{len(alerts)} alerts",
                )

        if any(result.sent for result in report.results):
            for alert in alerts:
                self._published[self._verdict_key(alert)] = self._verdict_shape(alert)

    # ----------------------------------------------------------------- storage

    async def _record(self, alert, chat_id: str, text: str, result: SendResult) -> None:
        if self.store is None:
            return
        try:
            await self.store.record_alert(
                chat_id=chat_id,
                text=text[:4000],
                news_id=alert.news_id,
                event_id=alert.event_id,
                analysis_id=alert.analysis_id,
                asset=alert.impact.asset,
                kind="update" if alert.is_update else "alert",
                message_id=result.message_id,
                sent=result.sent,
                error=result.error or None,
            )
        except Exception:
            # A storage failure must not lose an alert that already went out.
            log.exception("could not record alert for %s", alert.impact.asset)

    async def _record_raw(
        self, chat_id: str, text: str, result: SendResult, *, kind: str, asset: str = ""
    ) -> None:
        if self.store is None:
            return
        try:
            await self.store.record_alert(
                chat_id=chat_id, text=text[:4000], asset=asset[:60], kind=kind,
                message_id=result.message_id, sent=result.sent,
                error=result.error or None,
            )
        except Exception:
            log.exception("could not record %s message", kind)

    # -------------------------------------------------------------- plain text

    async def notify(self, text: str, *, kind: str = "notice") -> list[SendResult]:
        """Send an operational message (startup, shutdown, key exhaustion).

        Separate from :meth:`publish` because these bypass the score threshold —
        "every Gemini key is dead" is not an alert about the market, and holding
        it back because it scored zero would be exactly wrong.
        """
        if not self.settings.telegram.configured:
            log.info("telegram not configured; notice suppressed: %s", text[:120])
            return []

        results: list[SendResult] = []
        for chat_id in self.settings.telegram.chat_ids:
            result = await self.client.send(chat_id, text, silent=kind == "notice")
            results.append(result)
            await self._record_raw(chat_id, text, result, kind=kind)
        return results

    def forget(self) -> None:
        """Clear repeat-suppression memory. Used by tests and ``/reset``."""
        self._published.clear()


def urgency_of(score: float, magnitude: Magnitude) -> str:
    """Fallback urgency for an alert built outside the pipeline.

    The pipeline sets this itself; this exists so a hand-built alert (a replay, a
    test send) still renders with a sensible header emoji.
    """
    if score >= 85 or magnitude is Magnitude.EXTREME:
        return "critical"
    if score >= 70 or magnitude is Magnitude.HIGH:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


__all__ = ["DIGEST_THRESHOLD", "PublishReport", "Publisher", "urgency_of"]
