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
                              timeout: int = 300):
    """Planner backed by a LOCAL Ollama model (e.g. the fine-tuned GGUF pulled from HF).

    Uses the /api/chat endpoint with format=json so output is always syntactically
    valid JSON — we then score schema/op/canon/recovery to measure the fine-tune.
    """
    import urllib.request

    def planner(dirty_df, *_):
        profile = profile_dataframe(dirty_df)
        user = build_user_prompt(profile, dirty_df)
        payload = {
            "model": model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": 4000},
        }
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
        return plan
    return planner


if __name__ == "__main__":  # one-example smoke test
    import random
    from training.generate import make_example

    ex = make_example(random.Random(1))
    plan = make_ollama_planner()(ex["dirty_df"])
    print(json.dumps(plan, indent=2, ensure_ascii=False)[:1500])
