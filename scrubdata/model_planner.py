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


if __name__ == "__main__":  # one-example smoke test
    import random
    from training.generate import make_example

    ex = make_example(random.Random(1))
    plan = make_ollama_planner()(ex["dirty_df"])
    print(json.dumps(plan, indent=2, ensure_ascii=False)[:1500])
