"""Capture a raw v6 model plan LOCALLY (Ollama Q8_0 GGUF) for a Raha dataset.

Mirrors the Modal capture composition (scripts/modal_eval_v5.py --capture):
make_batched_planner(base, batch_size=4), greedy, no grounded wrapper, no union —
verification/union happen downstream (eval/raha_table.py, eval/precision_curve.py).
DISCLOSED deltas vs the Modal captures: (1) Q8_0 GGUF on local Ollama instead of the
bf16 merged adapter on A100 — quantization may shift individual mappings; (2) Ollama
format=json instead of generate(suppress_tokens=[151657,151658]) — both exist solely
to block the degenerate <tool_call> first token (without either, generation loops).

Prereq: ollama pull hf.co/ricalanis/scrubdata-qwen3-4b-v6-q8:Q8_0
        ollama create scrubdata-ft -f notebooks/Modelfile

    uv run python -m eval.capture_plan_local --dataset beers
Writes eval/results/v6_<dataset>_raw_plan_localq8.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scrubdata.model_planner import _extract_json, make_batched_planner

from .run_real_multi import _raha_pair


def make_json_constrained_planner(model: str, host: str = "http://localhost:11434",
                                  timeout: int = 600):
    """Local Ollama planner with format=json (grammar-constrained decoding)."""
    import urllib.request

    from scrubdata.profiler import profile_dataframe
    from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt

    def planner(dirty_df, *_):
        user = build_user_prompt(profile_dataframe(dirty_df), dirty_df)
        payload = {
            "model": model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": 2000, "num_ctx": 16384},
        }
        req = urllib.request.Request(
            host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"  batch failed: {str(e)[:80]}", flush=True)
            return {"__error__": str(e)[:120]}
        plan = _extract_json(out)
        if plan is None:
            print(f"  batch returned no JSON: {out[:80]!r}", flush=True)
            return {"__error__": "no_json"}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan
    return planner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="scrubdata-ft")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    dirty, _clean = _raha_pair(args.dataset)   # same table the scorer sees
    print(f"capturing plan: {args.dataset} ({len(dirty)} rows x {dirty.shape[1]} cols)",
          flush=True)
    t0 = time.time()
    plan = make_batched_planner(make_json_constrained_planner(args.model, timeout=args.timeout),
                                batch_size=4)(dirty)
    dt = time.time() - t0
    n_ops = sum(len(c.get("operations", [])) for c in plan.get("columns", []))
    print(f"done in {dt:.0f}s — {len(plan.get('columns', []))} columns, {n_ops} ops")

    out = (Path(__file__).resolve().parent / "results"
           / f"v6_{args.dataset}_raw_plan_localq8.json")
    json.dump(plan, open(out, "w"), indent=1)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
