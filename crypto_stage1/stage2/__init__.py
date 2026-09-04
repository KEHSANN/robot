"""Stage 2: generated narrative / output pipeline.

Stage 2 currently sits behind the same package boundary as Stage 0 and Stage 1
so the long-term structure already has a dedicated home.
"""

from __future__ import annotations

from .pipeline import Stage2Pipeline, run_stage2

__all__ = ["Stage2Pipeline", "run_stage2"]
