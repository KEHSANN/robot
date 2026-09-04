"""The feedback loop: prediction -> observation -> outcome -> model performance.

Everything upstream of this package produces claims. This is what checks them, and
it is what makes the rest of the system improvable rather than merely opinionated:
without it, a model that is confidently wrong is indistinguishable from one that is
right, and the panel would never get any cheaper or better.

Three pieces, deliberately separable so each can be tested without the others:

- :mod:`feedback.prices` — reads spot and historical prices from a public
  exchange. Knows that a baseline must come from the minute the news broke, and
  that not every asset a model names has a price at all.
- :mod:`feedback.score` — turns a prediction plus its observations into a score.
  This is where the definitions that decide whether the system looks good live, so
  they are written to be hard to flatter.
- :mod:`feedback.observe` — the loop that samples due horizons, resolves what can
  be resolved, and attributes the result to the models that produced it.

:mod:`feedback.score` is pure: no clock, no network, no database. That is why the
scoring rules can be tested exactly, which matters more here than anywhere else in
the pipeline.
"""

from feedback.observe import (
    BASELINE_DEADLINE,
    HORIZON_GRACE,
    ObserveStats,
    Observer,
    observe_once,
)
from feedback.prices import (
    ALIASES,
    PEGGED,
    UNPRICEABLE,
    Candle,
    PriceSource,
    PriceUnavailable,
    is_priceable,
    normalise_asset,
    pct_change,
)
from feedback.score import (
    DIRECTION_WEIGHT,
    Scored,
    combined_score,
    direction_of,
    expected_range,
    magnitude_error,
    performance_summary,
    score_prediction,
)

__all__ = [
    "ALIASES",
    "BASELINE_DEADLINE",
    "DIRECTION_WEIGHT",
    "HORIZON_GRACE",
    "PEGGED",
    "UNPRICEABLE",
    "Candle",
    "ObserveStats",
    "Observer",
    "PriceSource",
    "PriceUnavailable",
    "Scored",
    "combined_score",
    "direction_of",
    "expected_range",
    "is_priceable",
    "magnitude_error",
    "normalise_asset",
    "observe_once",
    "pct_change",
    "performance_summary",
    "score_prediction",
]
