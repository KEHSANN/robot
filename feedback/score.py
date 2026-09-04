"""Scoring a prediction against what the price actually did.

This is the module that decides whether the system is any good, so the definitions
here are the ones that matter most. Four choices, each of which could be made
wrongly in a way that flatters the models:

**Direction is scored with a deadband.** A prediction of BULLISH followed by a
+0.02% drift is not a correct call, it is noise that happened to have a sign.
Without the deadband every model scores ~50% on direction by chance and the metric
says nothing. ``FEEDBACK_DEADBAND_PCT`` is the width of "didn't move".

**A NEUTRAL call is a real claim and is scored as one.** It is correct exactly
when the price stayed inside the deadband. Treating NEUTRAL as unscoreable would
let a model score perfectly by never committing.

**Magnitude error is measured against the range, not the midpoint.** The models
were asked for a range, so a move that lands anywhere inside it is a hit with zero
error. Scoring against the midpoint would penalise a forecast that was correct as
stated, and would quietly reward narrow ranges over honest ones.

**Direction dominates the score.** A call that got the direction right and the
size wrong is useful; a call that got the size right and the direction wrong is
worse than silence, because it reads as a reason to take the losing side. So
direction is most of the score and magnitude accuracy adjusts it, rather than the
two being averaged.

The horizon result is kept separately: ``best_horizon_minutes`` records where the
move actually matched the forecast, which is how the system learns whether its
15-minute calls are really 6-hour calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from services.config import FeedbackSettings, settings as global_settings
from services.types import Direction, Outcome

log = logging.getLogger(__name__)

#: Weight of the direction call in the final score. The remainder is how close
#: the size was, which only pays out when the direction was right.
DIRECTION_WEIGHT = 0.65


@dataclass
class Scored:
    """A prediction's result, with the parts kept visible.

    The intermediate numbers are carried rather than collapsed because a summary
    that says only "score 0.4" cannot answer the question worth asking: was the
    direction wrong, or was the direction right and the size badly off?
    """

    direction_correct: bool
    actual_pct: float
    expected_pct: float
    magnitude_error: float
    score: float
    best_horizon_minutes: int
    #: Whether the move exceeded the deadband at all.
    moved: bool
    #: pct change at each observed horizon, for the outcome detail.
    by_horizon: dict[int, float]

    def as_outcome(self, prediction_id: int, **detail) -> Outcome:
        return Outcome(
            prediction_id=prediction_id,
            direction_correct=self.direction_correct,
            actual_pct=self.actual_pct,
            expected_pct=self.expected_pct,
            magnitude_error=self.magnitude_error,
            score=self.score,
            best_horizon_minutes=self.best_horizon_minutes,
            detail={
                "by_horizon": {str(key): round(value, 4)
                               for key, value in self.by_horizon.items()},
                "moved": self.moved,
                **detail,
            },
        )


def expected_range(low: float, high: float, direction: Direction) -> tuple[float, float]:
    """The forecast as a signed, ordered range.

    Stages report magnitudes as positive numbers with the sign carried by
    direction, but some models return them already signed. Both are accepted,
    because rejecting the signed form would silently drop that model's forecasts.
    """
    magnitude_low, magnitude_high = sorted((abs(low), abs(high)))
    if direction is Direction.BEARISH:
        return (-magnitude_high, -magnitude_low)
    if direction is Direction.BULLISH:
        return (magnitude_low, magnitude_high)
    # NEUTRAL and MIXED do not claim a side, so the range is symmetric.
    return (-magnitude_high, magnitude_high)


def direction_of(pct: float, deadband: float) -> Direction:
    """What the price actually did, at the resolution the deadband allows."""
    if pct > deadband:
        return Direction.BULLISH
    if pct < -deadband:
        return Direction.BEARISH
    return Direction.NEUTRAL


def magnitude_error(actual_pct: float, low: float, high: float) -> float:
    """Distance from the actual move to the nearest edge of the forecast range.

    Zero inside the range: the forecast said "between 3% and 7%", and 5% is not a
    less correct answer than 4% would have been.
    """
    if low <= actual_pct <= high:
        return 0.0
    return min(abs(actual_pct - low), abs(actual_pct - high))


def score_prediction(
    *,
    direction: Direction | str,
    expected_low: float,
    expected_high: float,
    horizon_minutes: int,
    observations: dict[int, float],
    config: FeedbackSettings | None = None,
) -> Scored | None:
    """Score one prediction. ``None`` when nothing has been observed yet.

    ``observations`` maps an offset in minutes to the percentage move at that
    offset. The prediction's own horizon is what gets scored; the other offsets
    are kept for the horizon diagnostic.
    """
    settings = config or global_settings.feedback
    deadband = max(0.0, settings.direction_deadband_pct)

    if not observations:
        return None

    called = direction if isinstance(direction, Direction) else Direction.parse(direction)
    low, high = expected_range(expected_low, expected_high, called)

    # Score at the prediction's own horizon when it was observed. Otherwise the
    # nearest observed offset, so a prediction is not left unscored because a
    # single sample was missed.
    offset = _closest_offset(observations, horizon_minutes)
    actual = observations[offset]

    actual_direction = direction_of(actual, deadband)
    moved = actual_direction is not Direction.NEUTRAL

    if called in (Direction.NEUTRAL, Direction.MIXED):
        # The claim was "this does not move it", which is right when it did not.
        correct = not moved
    else:
        correct = actual_direction is called

    error = magnitude_error(actual, low, high)
    expected_midpoint = (low + high) / 2

    return Scored(
        direction_correct=correct,
        actual_pct=round(actual, 4),
        expected_pct=round(expected_midpoint, 4),
        magnitude_error=round(error, 4),
        score=round(combined_score(correct, error, low, high), 4),
        best_horizon_minutes=_best_horizon(
            observations, low, high, called, deadband, horizon_minutes
        ),
        moved=moved,
        by_horizon=dict(observations),
    )


def combined_score(direction_correct: bool, error: float, low: float, high: float) -> float:
    """One number in ``0..1``, weighted so direction dominates.

    A wrong direction is capped well below a right one no matter how close the
    size was, because acting on a wrong direction loses money that a wrong size
    only leaves on the table.
    """
    if not direction_correct:
        # A wrong call that at least got the size right is marginally less bad —
        # it means the event was correctly sized and misread, not misunderstood.
        tolerance = max(1.0, abs(high - low), abs(high))
        closeness = max(0.0, 1.0 - error / (tolerance * 3))
        return round((1.0 - DIRECTION_WEIGHT) * closeness * 0.5, 4)

    # Scale the size penalty by the width of what was forecast: being 2% off a
    # 10% call is a better forecast than being 2% off a 0.5% call.
    tolerance = max(0.5, abs(high - low), abs(high) * 0.5)
    accuracy = max(0.0, 1.0 - error / (tolerance * 2))
    return round(DIRECTION_WEIGHT + (1.0 - DIRECTION_WEIGHT) * accuracy, 4)


def _closest_offset(observations: dict[int, float], target: int) -> int:
    """The observed offset nearest the prediction's horizon."""
    return min(observations, key=lambda offset: (abs(offset - target), offset))


def _best_horizon(
    observations: dict[int, float],
    low: float,
    high: float,
    called: Direction,
    deadband: float,
    horizon_minutes: int,
) -> int:
    """Where the move best matched the forecast.

    This is the horizon diagnostic: if 15-minute calls keep scoring best at 6
    hours, the models are right about the event and wrong about how fast the
    market prices it, which is a different correction from being wrong outright.

    Ties are broken toward the horizon that was actually called. Several offsets
    landing inside the forecast range is the normal case for a good call, and
    breaking the tie by earliest offset would report a well-timed prediction as
    mistimed — which would turn the diagnostic into noise pointing one direction.
    """
    if not observations:
        return 0

    def penalty(offset: int) -> tuple[float, int, int]:
        actual = observations[offset]
        wrong_side = (
            called in (Direction.BULLISH, Direction.BEARISH)
            and direction_of(actual, deadband) is not called
        )
        return (
            # A horizon on the wrong side is never the best match, whatever its size.
            magnitude_error(actual, low, high) + (1000.0 if wrong_side else 0.0),
            abs(offset - horizon_minutes),
            offset,
        )

    return min(observations, key=penalty)


def performance_summary(rows: list[dict]) -> list[dict]:
    """Rank model performance rows by hit rate, with a minimum sample size.

    A model with three predictions and three hits is not a 100% model. Ranking
    without the floor would put whichever model has answered least at the top,
    which is precisely the wrong signal to act on.
    """
    ranked = []
    for row in rows:
        total = int(row.get("predictions") or 0)
        if total <= 0:
            continue
        correct = int(row.get("direction_correct") or 0)
        ranked.append(
            {
                **row,
                "hit_rate": correct / total,
                # Wilson-style shrink toward 50% so small samples sort sensibly
                # without being excluded outright.
                "adjusted": (correct + 2) / (total + 4),
            }
        )

    ranked.sort(key=lambda row: (row["adjusted"], row["predictions"]), reverse=True)
    return ranked


__all__ = [
    "DIRECTION_WEIGHT",
    "Scored",
    "combined_score",
    "direction_of",
    "expected_range",
    "magnitude_error",
    "performance_summary",
    "score_prediction",
]
