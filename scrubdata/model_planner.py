"""LLM planner backend via Ollama Cloud (`oll`).

Runs a *vanilla* (un-fine-tuned) model as the planner so we can measure it on the
eval harness — the disciplined "is a fine-tune even needed, and how big is the gap"
check. Same interface as scrubdata.mock_plan: callable(dirty_df) -> plan dict.

Cloud hosts large models, so this is mainly the BIG-MODEL CEILING / teacher reference
(and a task sanity check), not the ≤4B target we fine-tune locally.
"""

from __future__ import annotations

import json
import subprocess

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .profiler import profile_dataframe


def _extract_json(text: str) -> dict | None:
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def make_ollama_planner(model: str = "glm-5.1", max_tokens: int = 8000,
                        timeout: int = 300):
    """Return a planner(dirty_df, *_ ) -> plan dict backed by an Ollama Cloud model."""
    def planner(dirty_df, *_):
        profile = profile_dataframe(dirty_df)
        user = build_user_prompt(profile, dirty_df)
        try:
            r = subprocess.run(
                ["oll", "--model", model, "--system", SYSTEM_PROMPT,
                 "--max-tokens", str(max_tokens), "--temperature", "0"],
                input=user, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return {"__error__": "call_failed"}
        plan = _extract_json(r.stdout)
        if plan is None:
            return {"__error__": "no_json", "raw": r.stdout[:200]}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan
    return planner


def make_local_ollama_planner(model: str, host: str = "http://localhost:11434",
                              timeout: int = 300, pair_profiles: bool = False):
    """Planner backed by a LOCAL Ollama model (e.g. the fine-tuned GGUF pulled from HF).

    Uses the /api/chat endpoint with format=json so output is always syntactically
    valid JSON — we then score schema/op/canon/recovery to measure the fine-tune.
    `pair_profiles` (WS2, off by default): prompt lists evidence-backed repair
    candidates and the output is constrained to them (constrain_plan).
    """
    import os
    import urllib.request

    # On some CUDA GPUs (verified: the A100 serving path) the non-thinking GGUF
    # degenerates into a <tool_call> loop even with flash-attention off; constraining
    # the API call to format=json grammar-kills the loop and yields clean JSON ~2x
    # faster. Our served Modelfile is non-thinking (no <think> prefix), so format=json
    # is safe. Env-gated so a thinking-template local run (where it'd reject the
    # <think> prefix) keeps the _extract_json path. Set SCRUBDATA_OLLAMA_FORMAT_JSON=1.
    force_json = os.environ.get("SCRUBDATA_OLLAMA_FORMAT_JSON") == "1"

    def planner(dirty_df, *_):
        profile = profile_dataframe(dirty_df)
        pairs = None
        if pair_profiles:
            from .pair_profile import pairs_for_df
            pairs = pairs_for_df(dirty_df)
        user = build_user_prompt(profile, dirty_df, candidate_pairs=pairs)
        # Default (no format=json): Qwen3-Instruct emits an empty <think></think> prefix
        # that a strict JSON constraint rejects; _extract_json pulls the plan out. The
        # force_json path (above) is for serving GPUs that degenerate without it.
        payload = {
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            # num_ctx must hold the (large) real-data profile + output; Ollama defaults
            # to ~4k which 400s on wide tables.
            "options": {"temperature": 0, "num_predict": 2000, "num_ctx": 16384},
        }
        if force_json:
            payload["format"] = "json"
        req = urllib.request.Request(
            host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())["message"]["content"]
        except Exception as e:
            return {"__error__": str(e)[:120]}
        plan = _extract_json(out)
        if plan is None:
            return {"__error__": "no_json", "raw": out[:200]}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        if pairs is not None:
            from .pair_profile import constrain_plan
            plan = constrain_plan(plan, pairs)
        return plan
    return planner


def make_batched_planner(base_planner, batch_size: int = 6):
    """Agentic wrapper: plan a WIDE table column-batch by column-batch.

    Aggregation (value_counts) makes the prompt invariant to ROW count; batching makes
    it invariant to COLUMN count. Each model call sees only `batch_size` columns (a small
    profile), so the planner scales to arbitrarily large/wide tables. Table-level ops
    (dedup, drop empty) are deterministic. `base_planner` is any callable(df)->plan.
    """
    from .profiler import profile_dataframe

    def planner(dirty_df, *_):
        prof = profile_dataframe(dirty_df)
        empty_cols = prof["empty_columns"]
        table_ops = []
        if prof["n_empty_rows"]:
            table_ops.append({"op": "drop_empty_rows", "rationale": "Fully-empty row(s)."})
        if empty_cols:
            table_ops.append({"op": "drop_empty_columns", "columns": empty_cols,
                              "rationale": "Column(s) with no data."})
        if prof["n_exact_duplicate_rows"]:
            table_ops.append({"op": "drop_exact_duplicates",
                              "rationale": "Exact duplicate row(s)."})

        work = [c for c in dirty_df.columns if c not in empty_cols]
        columns, flags = [], []
        for i in range(0, len(work), batch_size):
            sub = dirty_df[work[i:i + batch_size]]
            p = base_planner(sub)
            if isinstance(p, dict):
                columns.extend(p.get("columns", []))
                flags.extend(p.get("flags", []))
        return {
            "dataset_summary": f"{prof['n_rows']} rows × {prof['n_cols']} columns "
                               f"(planned in {((len(work) - 1) // batch_size) + 1} batches).",
            "table_operations": table_ops, "columns": columns, "flags": flags,
        }
    return planner


if __name__ == "__main__":  # one-example smoke test
    import random
    from training.generate import make_example

    ex = make_example(random.Random(1))
    plan = make_ollama_planner()(ex["dirty_df"])
    print(json.dumps(plan, indent=2, ensure_ascii=False)[:1500])
