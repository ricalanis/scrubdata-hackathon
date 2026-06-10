"""The active planner used by the app/server/CLI — the verified union planner.

`get_planner()` returns the WS1 pipeline when a model backend is configured
(env `SCRUBDATA_MODEL`, e.g. a local Ollama model id), otherwise the deterministic
grounded heuristic. The model pipeline is:

    grounded model plan -> per-entry verifier (tau, default 0.5; dropped merges
    become review flags) -> union with the grounded heuristic's mappings
    (model wins per surface form)

Measured on hospital (509 real errors): 0.905 precision @ 0.413 coverage — the
verifier kills the model's low-confidence merges (model alone gated: 0.993 @ 0.287),
the heuristic union buys back coverage. The heuristic also still covers any
column-batch the model errors/times out on, so the app is never left without a plan
(the trust contract from PRODUCT.md).

    SCRUBDATA_MODEL=scrubdata-ft-v6 uv run server.py   # use the fine-tune (local Ollama)
    SCRUBDATA_TAU=0.7 ...                              # stricter verifier (optional)
    uv run server.py                                   # grounded heuristic (always works)
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

    # WS2 pair-profiles: candidate-constrained canonicalization (opt-in; ships
    # default-on only if it clears the WS1 gate on hospital)
    pair_profiles = os.environ.get("SCRUBDATA_PAIR_PROFILES") == "1"
    raw = make_local_ollama_planner(model, pair_profiles=pair_profiles)

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

    # RACOON: ground the model's canonicalization against reference taxonomies — the
    # model never free-generates a canonical for a reference-typed column.
    from scrubdata.grounded import make_grounded_planner
    grounded = make_grounded_planner(batched)

    from scrubdata.verifier import union_plans, verify_plan
    tau = float(os.environ.get("SCRUBDATA_TAU", "0.5"))

    def verified_union(df, *_):
        plan = verify_plan(df, grounded(df), tau=tau)
        plan = union_plans(plan, mock_plan(df))
        plan["_generated_by"] = f"verified-union(model:{model}, tau={tau})"
        return plan

    return verified_union
