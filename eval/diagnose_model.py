"""Diagnose vanilla-model failures: truncation vs genuine schema violation.

Runs N examples through an Ollama Cloud model, categorizing each output:
  empty / no_json / truncated / json_but_schema_invalid / valid
and reading `oll`'s stderr token counts to detect output hitting the cap.

    uv run eval/diagnose_model.py --n 12 --model glm-5.1 --max-tokens 8000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import Counter

from jsonschema import Draft202012Validator

from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
from scrubdata.profiler import profile_dataframe
from training.generate import make_example

from .metrics import PLAN_SCHEMA

_V = Draft202012Validator(PLAN_SCHEMA)
_TOK = re.compile(r"out\s+(\d+)\s*tok", re.I)


def _call(user: str, model: str, max_tokens: int):
    r = subprocess.run(
        ["oll", "--model", model, "--system", SYSTEM_PROMPT,
         "--max-tokens", str(max_tokens), "--temperature", "0"],
        input=user, capture_output=True, text=True, timeout=300)
    out_tok = None
    m = _TOK.search(r.stderr or "")
    if m:
        out_tok = int(m.group(1))
    return r.stdout, out_tok


def _categorize(out: str, out_tok: int | None, max_tokens: int):
    s = out.strip()
    if not s:
        return "empty", None
    i, j = s.find("{"), s.rfind("}")
    if i == -1:
        return "no_json", None
    near_cap = out_tok is not None and out_tok >= max_tokens - 50
    if j < i:
        return ("truncated" if near_cap else "no_close_brace"), None
    try:
        plan = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return ("truncated" if near_cap else "malformed_json"), None
    errs = sorted(_V.iter_errors(plan), key=lambda e: e.path)
    if not errs:
        return "valid", None
    return "schema_invalid", errs[0].message[:90]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", type=str, default="glm-5.1")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cats = Counter()
    print(f"Diagnosing {args.model} @ max_tokens={args.max_tokens} on {args.n} examples\n")
    for k in range(args.n):
        ex = make_example(rng)
        user = build_user_prompt(profile_dataframe(ex["dirty_df"]), ex["dirty_df"])
        out, out_tok = _call(user, args.model, args.max_tokens)
        cat, detail = _categorize(out, out_tok, args.max_tokens)
        cats[cat] += 1
        print(f"  ex{k:2d}: {cat:<16} out_tok={out_tok}"
              + (f"  [{detail}]" if detail else ""))

    print("\nBreakdown:", dict(cats))
    valid = cats.get("valid", 0)
    trunc = cats.get("truncated", 0)
    print(f"valid={valid}/{args.n} ({valid/args.n:.0%}) | truncated={trunc} "
          f"| schema_invalid={cats.get('schema_invalid', 0)}")


if __name__ == "__main__":
    main()
