"""W1.c — ≤32B zero-label repair SCALING ARM (multi-family, zero-shot).

Runs vanilla (NOT fine-tuned) 20–31B open-weights models as the raw planner inside
the EXACT hospital pipeline the 4B fine-tune gate used, so the points are comparable:

    batched raw planner (batch_size=4, SYSTEM_PROMPT + build_user_prompt, temp 0)
      -> verify_plan(tau=0.5)                    [gated point]
      -> union_plans(verified, mock_plan(dirty)) [shipped active.py composition]

Provenance match (verified in-session): re-scoring the captured v6 fine-tune plan
(eval/results/v6_hospital_raw_plan.json) through this scorer reproduces the prior
measurement exactly — gated 0.993/0.287, union 0.905/0.413, kept=132 dropped=38.
The v6 capture (scripts/modal_eval_v5.py) did NOT apply the grounded wrapper to the
model plan; we match that protocol (the grounded heuristic enters via the union).

Models are served via Ollama Cloud through the local daemon's cloud proxy
(`ollama pull <id>-cloud`): hosted inference for speed; all are locally deployable
in principle (≤32B open weights). The local GPU is never used.

    uv run python -m eval.scaling_arm
    uv run python -m eval.scaling_arm --models gpt-oss:20b-cloud --budget-s 1200
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from scrubdata.executor import apply_plan
from scrubdata.model_planner import _extract_json, make_batched_planner
from scrubdata.planner import mock_plan
from scrubdata.profiler import profile_dataframe
from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
from scrubdata.verifier import union_plans, verify_plan

from .precision_curve import _repairs_only
from .run_real import _ensure_data, _load
from .run_real_multi import score as _cn_score

RESULTS = Path(__file__).resolve().parent / "results"

# model id -> (params_b from the tag, family)
DEFAULT_MODELS = {
    "gpt-oss:20b-cloud": (20, "openai/gpt-oss"),
    "devstral-small-2:24b-cloud": (24, "mistral/devstral"),
    "nemotron-3-nano:30b-cloud": (30, "nvidia/nemotron"),
    "gemma4:31b-cloud": (31, "google/gemma"),
}

# the shipped 4B fine-tune point — copied, not re-run (provenance: commit aa48108,
# eval/results/v6_hospital_raw_plan.json + union_gate_point.json). kept/dropped
# re-derived in-session from the captured plan artifact at tau=0.5.
PRIOR_4B_ROW = {
    "model": "scrubdata-ft-v6 (Qwen3-4B fine-tune)",
    "params_b": 4, "family": "qwen/qwen3 (fine-tuned)",
    "gated_prec": 0.993, "gated_cov": 0.287,
    "union_prec": 0.905, "union_cov": 0.413,
    "validity": None, "runtime_s": None,
    "verifier_kept": 132, "verifier_dropped": 38,
    "provenance": "prior measurement (commit aa48108; Modal A100 capture; "
                  "validity/runtime not recorded at capture)",
}


def make_cloud_planner(model: str, host: str = "http://localhost:11434",
                       timeout: int = 300, num_predict: int = 4000):
    """Raw zero-shot planner via the daemon's Ollama Cloud proxy — same /api/chat
    contract as scrubdata.model_planner.make_local_ollama_planner, with a larger
    num_predict so reasoning-channel models are not budget-truncated. Counts calls
    and JSON-valid plans for the validity-rate measurement."""
    stats = {"calls": 0, "valid": 0, "errors": []}

    def planner(dirty_df, *_):
        stats["calls"] += 1
        profile = profile_dataframe(dirty_df)
        user = build_user_prompt(profile, dirty_df)
        payload = {
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": num_predict,
                        "num_ctx": 16384},
        }
        req = urllib.request.Request(
            host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())["message"]["content"]
        except Exception as e:
            stats["errors"].append(str(e)[:120])
            return {"__error__": str(e)[:120]}
        plan = _extract_json(out)
        if plan is None:
            stats["errors"].append("no_json:" + out[:80].replace("\n", " "))
            return {"__error__": "no_json"}
        stats["valid"] += 1
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan

    return planner, stats


def kept_dropped(verified_plan: dict) -> tuple[int, int]:
    k = d = 0
    for col in verified_plan.get("columns", []):
        for op in col.get("operations", []):
            v = op.get("_verified")
            if v:
                k += v["kept"]
                d += v["dropped"]
    return k, d


def score_point(dirty, clean, raw_plan: dict, tau: float) -> dict:
    """The v6 gate protocol at one tau: verify -> (gated | union) -> repairs-only
    -> apply -> churn-neutral score."""
    verified = verify_plan(dirty, raw_plan, tau=tau)
    k, d = kept_dropped(verified)
    out = {"verifier_kept": k, "verifier_dropped": d}
    for label, plan in (("gated", verified),
                        ("union", union_plans(verified, mock_plan(dirty)))):
        cleaned, _ = apply_plan(dirty, _repairs_only(plan))
        m = _cn_score(dirty, clean, cleaned)
        out[f"{label}_prec"] = round(m["precision"], 3)
        out[f"{label}_cov"] = round(m["recall"], 3)
        out[f"{label}_changed"] = m["_changed"]
        out[f"{label}_fixed"] = m["_fixed"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help="comma-separated cloud model ids")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--budget-s", type=float, default=1200,
                    help="per-model wall-clock budget (default 20 min)")
    ap.add_argument("--num-predict", type=int, default=4000,
                    help="per-call generation budget (reasoning models need ~8000: "
                         "thinking tokens count against it)")
    ap.add_argument("--out", type=str, default=str(RESULTS / "scaling_arm.json"))
    args = ap.parse_args()

    _ensure_data()
    dirty, clean = _load()
    rows = [dict(PRIOR_4B_ROW)]

    for model in args.models.split(","):
        params_b, family = DEFAULT_MODELS.get(model, (None, model.split(":")[0]))
        print(f"\n=== {model} ({family}, {params_b}B) — hospital, 509 real errors ===",
              flush=True)
        raw, stats = make_cloud_planner(model, num_predict=args.num_predict)
        batched = make_batched_planner(raw, batch_size=4)
        t0 = time.time()
        try:
            raw_plan = batched(dirty)
        except Exception as e:                       # never lose the arm to one model
            raw_plan = {"__error__": str(e)[:200]}
        runtime = round(time.time() - t0, 1)
        validity = round(stats["valid"] / stats["calls"], 3) if stats["calls"] else 0.0
        slug = model.replace(":", "_").replace("/", "_")
        plan_path = RESULTS / f"scaling_{slug}_hospital_raw_plan.json"
        json.dump(raw_plan, open(plan_path, "w"))

        row = {"model": model, "params_b": params_b, "family": family,
               "validity": validity, "runtime_s": runtime,
               "n_calls": stats["calls"], "errors": stats["errors"][:5],
               "provenance": f"this run (zero-shot via Ollama Cloud proxy, "
                             f"num_predict={args.num_predict})",
               "raw_plan": str(plan_path.relative_to(RESULTS.parent.parent))}
        if isinstance(raw_plan, dict) and "__error__" not in raw_plan:
            row.update(score_point(dirty, clean, raw_plan, tau=args.tau))
        else:
            row.update({"gated_prec": None, "gated_cov": None,
                        "union_prec": None, "union_cov": None,
                        "verifier_kept": None, "verifier_dropped": None,
                        "note": "planner produced no plan"})
        if runtime > args.budget_s:
            row["note"] = f"exceeded {args.budget_s:.0f}s budget"
        rows.append(row)
        print(f"  validity {validity}  runtime {runtime}s  "
              f"gated {row.get('gated_prec')}/{row.get('gated_cov')}  "
              f"union {row.get('union_prec')}/{row.get('union_cov')}  "
              f"kept/dropped {row.get('verifier_kept')}/{row.get('verifier_dropped')}",
              flush=True)

    json.dump({"task": "W1.c scaling arm — hospital 509 real errors, tau=0.5, "
                       "v6 gate protocol (batched raw plan -> verify -> union)",
               "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nwritten: {args.out}")

    hdr = f"{'model':<38}{'B':>4}{'gated P/C':>15}{'union P/C':>15}{'valid':>7}{'kept/drop':>11}{'s':>7}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        gp = f"{r['gated_prec']}/{r['gated_cov']}" if r.get("gated_prec") is not None else "—"
        up = f"{r['union_prec']}/{r['union_cov']}" if r.get("union_prec") is not None else "—"
        kd = (f"{r['verifier_kept']}/{r['verifier_dropped']}"
              if r.get("verifier_kept") is not None else "—")
        v = r["validity"] if r.get("validity") is not None else "—"
        s = r["runtime_s"] if r.get("runtime_s") is not None else "—"
        print(f"{r['model']:<38}{r['params_b']:>4}{gp:>15}{up:>15}{v:>7}{kd:>11}{s:>7}")


if __name__ == "__main__":
    main()
