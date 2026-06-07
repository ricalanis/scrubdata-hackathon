"""Agent-trace capture for ScrubData.

Each cleaning run is an agent decision: profile (input) → cleaning plan (output) →
executed changes. We log those as JSONL so they can be published to the Hub as an
open agent trace (the "Sharing is Caring / Open trace" bonus quest), mirroring how
HF's own ml-intern auto-shares traces.

data/ is gitignored — traces are runtime artifacts, published separately via
scripts/publish_traces.py, not committed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .prompt import build_user_prompt, serialize_plan

DEFAULT_TRACE_PATH = Path("data/traces/scrubdata-traces.jsonl")


def build_trace(profile: dict, sample_df: pd.DataFrame, plan: dict,
                change_log: list[dict], *, model: str = "mock_planner",
                duration_ms: float | None = None, ts: float | None = None) -> dict:
    """One agent-trace record: the planner's input, output, and applied effect."""
    column_ops = [
        {"column": c["name"], "type": c.get("detected_semantic_type"),
         "ops": [o["op"] for o in c.get("operations", [])]}
        for c in plan.get("columns", [])
    ]
    return {
        "ts": ts if ts is not None else time.time(),
        "model": model,
        "n_rows": profile.get("n_rows"),
        "n_cols": profile.get("n_cols"),
        "agent_input": build_user_prompt(profile, sample_df),   # what the planner saw
        "agent_output_plan": serialize_plan(plan),              # the JSON plan it produced
        "table_operations": [o["op"] for o in plan.get("table_operations", [])],
        "column_operations": column_ops,
        "cells_changed": sum(e.get("cells_changed", 0)
                             for e in change_log if e.get("scope") == "column"),
        "duration_ms": duration_ms,
    }


def append_trace(record: dict, path: Path | str = DEFAULT_TRACE_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return p


def log_run(profile: dict, sample_df: pd.DataFrame, plan: dict, change_log: list[dict],
            *, model: str = "mock_planner", duration_ms: float | None = None,
            path: Path | str = DEFAULT_TRACE_PATH) -> dict:
    """Build + append a trace. Best-effort: callers should not let tracing break a run."""
    rec = build_trace(profile, sample_df, plan, change_log,
                      model=model, duration_ms=duration_ms)
    append_trace(rec, path)
    return rec
