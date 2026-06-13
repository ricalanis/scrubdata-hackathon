"""OpenBMB MiniCPM gate check — drops a zero-shot MiniCPM planner into the IDENTICAL
hospital verify(tau=0.5)+union gate the scaling arm uses, and scores it against the
shipped Qwen3-4B fine-tune bar (union 0.905@0.413).

Reuses eval/scaling_arm.py wholesale (make_cloud_planner hits /api/chat, so a LOCAL
ollama model id works through the same contract — no cloud proxy involved here). Does
not modify scaling_arm's existing rows; writes eval/results/minicpm_check.json.

    uv run python -m eval.minicpm_check
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scrubdata.model_planner import make_batched_planner

from .run_real import _ensure_data, _load
from .scaling_arm import (PRIOR_4B_ROW, RESULTS, make_cloud_planner, score_point)

MODEL = "hf.co/openbmb/MiniCPM3-4B-GGUF:Q4_K_M"
PARAMS_B = 4
FAMILY = "openbmb/minicpm3 (zero-shot, local)"
TAU = 0.5


def main() -> None:
    _ensure_data()
    dirty, clean = _load()

    print(f"=== {MODEL} ({FAMILY}, {PARAMS_B}B) — hospital, 509 real errors ===",
          flush=True)
    # local model, served by the same daemon at localhost:11434 via /api/chat;
    # bigger num_predict so a chatty 4B isn't truncated mid-JSON
    raw, stats = make_cloud_planner(MODEL, num_predict=4000)
    batched = make_batched_planner(raw, batch_size=4)
    t0 = time.time()
    try:
        raw_plan = batched(dirty)
    except Exception as e:
        raw_plan = {"__error__": str(e)[:200]}
    runtime = round(time.time() - t0, 1)
    validity = round(stats["valid"] / stats["calls"], 3) if stats["calls"] else 0.0

    plan_path = RESULTS / "scaling_minicpm3_4b_hospital_raw_plan.json"
    json.dump(raw_plan, open(plan_path, "w"))

    row = {"model": MODEL, "params_b": PARAMS_B, "family": FAMILY,
           "validity": validity, "runtime_s": runtime,
           "n_calls": stats["calls"], "errors": stats["errors"][:5],
           "provenance": "this run (ZERO-SHOT, local RTX 3060 Ti via ollama "
                         "hf.co/openbmb/MiniCPM3-4B-GGUF:Q4_K_M, num_predict=4000)",
           "raw_plan": str(plan_path.relative_to(RESULTS.parent.parent))}
    if isinstance(raw_plan, dict) and "__error__" not in raw_plan:
        row.update(score_point(dirty, clean, raw_plan, tau=TAU))
    else:
        row.update({"gated_prec": None, "gated_cov": None,
                    "union_prec": None, "union_cov": None,
                    "verifier_kept": None, "verifier_dropped": None,
                    "note": "planner produced no plan"})

    print(f"  validity {validity}  runtime {runtime}s  "
          f"gated {row.get('gated_prec')}/{row.get('gated_cov')}  "
          f"union {row.get('union_prec')}/{row.get('union_cov')}  "
          f"kept/dropped {row.get('verifier_kept')}/{row.get('verifier_dropped')}",
          flush=True)

    bar = PRIOR_4B_ROW
    out = {
        "task": "OpenBMB MiniCPM gate check — hospital 509 real errors, tau=0.5, "
                "v6 gate protocol (batched raw plan -> verify -> union). "
                "ZERO-SHOT MiniCPM vs FINE-TUNED Qwen3-4B bar.",
        "bar": {"model": bar["model"], "union_prec": bar["union_prec"],
                "union_cov": bar["union_cov"], "gated_prec": bar["gated_prec"],
                "gated_cov": bar["gated_cov"]},
        "rows": [dict(bar), row],
    }
    json.dump(out, open(RESULTS / "minicpm_check.json", "w"), indent=1)
    print("\nwritten:", RESULTS / "minicpm_check.json")


if __name__ == "__main__":
    main()
