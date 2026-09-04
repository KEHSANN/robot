"""Analysis stages 1-5 plus the final NVIDIA layer.

Stage 0 lives in its own package because event detection is substantial enough to
warrant one. These are single modules sharing the panel machinery in
:mod:`stages.base`, which is what makes them short: each one holds a prompt's
parser and the judgement about how to aggregate its answers, and nothing else.

Import order note: :mod:`stages.pipeline` imports every stage, so importing it
imports the whole chain. The stage modules themselves import only ``base`` and,
where they share aggregation, ``stage4`` — never each other otherwise.
"""

from stages.base import PanelRunner, StageContext, best_evidence, clamp, majority_of
from stages.pipeline import Alert, Pipeline, PipelineStats, Result
from stages.router import Escalation, Router

__all__ = [
    "Alert",
    "Escalation",
    "PanelRunner",
    "Pipeline",
    "PipelineStats",
    "Result",
    "Router",
    "StageContext",
    "best_evidence",
    "clamp",
    "majority_of",
]
