"""The active planner used by the app/server/CLI — model with heuristic fallback.

`get_planner()` returns the fine-tuned model planner when a backend is configured
(env `SCRUBDATA_MODEL`, e.g. a local Ollama model id), otherwise the deterministic
heuristic. The model handles the fuzzy canonicalization it was trained for; the
heuristic covers any column-batch the model errors/times out on — so the app is never
left without a plan (the trust contract from PRODUCT.md).

    SCRUBDATA_MODEL=scrubdata-ft-v4q8 uv run server.py   # use the fine-tune (local Ollama)
    uv run server.py                                     # heuristic only (always works)
"""

from __future__ import annotations

import os

from .planner import mock_plan


def get_planner():
    """Return a callable(df) -> plan dict: the model (if configured) else the heuristic."""
    model = os.environ.get("SCRUBDATA_MODEL")
    if not model:
        return mock_plan

    from .model_planner import make_local_ollama_planner, make_batched_planner

    raw = make_local_ollama_planner(model)

    def model_or_heuristic(df, *_):
        # per column-batch: try the model, fall back to the heuristic on any failure
        try:
            p = raw(df)
            if isinstance(p, dict) and "__error__" not in p and p.get("columns"):
                return p
        except Exception:
            pass
        return mock_plan(df)

    # batching makes it scale to wide tables (each call sees a few columns)
    batched = make_batched_planner(model_or_heuristic, batch_size=4)

    def tagged(df, *_):
        plan = batched(df)
        plan["_generated_by"] = f"model:{model}"
        return plan

    return tagged
