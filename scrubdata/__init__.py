"""ScrubData — hands-off tabular data cleaning.

Pipeline (see PRODUCT.md):
    profile  ->  plan  ->  execute  ->  report

`profiler` and `executor` are deterministic and final. `planner` is currently a
heuristic MOCK standing in for the fine-tuned ≤4B model; swap `mock_plan` for the
model call without touching the rest of the pipeline.
"""

from .profiler import profile_dataframe
from .planner import mock_plan
from .executor import apply_plan
from .report import render_report

__all__ = ["profile_dataframe", "mock_plan", "apply_plan", "render_report"]
