"""Stage 1: consensus over candidate event assignments."""

from __future__ import annotations

from .consensus import majority_consensus
from .runner import run_stage1
from .schemas import Stage0Output, Stage1Decision, Stage1Result

__all__ = [
    "Stage0Output",
    "Stage1Decision",
    "Stage1Result",
    "majority_consensus",
    "run_stage1",
]
