"""GRPO episode corpus: error-dense windows from TRAIN-side paired sources.

Each episode = {messages: [system,user] (the standard planner prompt for a small
window of a real paired table), dirty_csv, clean_csv}. NO derived gold plan — the
RL reward is computed by EXECUTING the model's plan against the clean slice
(churn-neutral F1 − damage), which is exactly the published RLVR recipe with our
executor as the verifier. Eval-side sources (flights/rayyan/ed2/tt/zeroed/dgov)
are NEVER episode material.

    uv run python -m training.build_grpo_episodes --n 600 --out data/grpo_episodes.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import random

from scrubdata.profiler import profile_dataframe
from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt

from .real_data import _cell_equal, canonicalizable_columns, fetch_pair

TRAIN_SOURCES = ["hospital", "beers", "movies_1", "cleanml_company",
                 "fodors_zagats", "gidcl_imdb"]


def windows(name: str, rng: random.Random, n_per: int,
            min_rows=20, max_rows=70, max_cols=5):
    dirty, clean = fetch_pair(name, keep_raw=True)
    n = min(len(dirty), len(clean))
    cols = canonicalizable_columns(dirty, clean)
    if not cols:
        return
    diff_rows = sorted({i for j in cols for i in range(n)
                        if not _cell_equal(dirty.iat[i, j], clean.iat[i, j])})
    if not diff_rows:
        return
    made = tries = 0
    while made < n_per and tries < n_per * 8:
        tries += 1
        k = rng.randint(min_rows, min(max_rows, n))
        anchor = rng.choice(diff_rows)
        start = max(0, min(anchor - rng.randint(0, k - 1), n - k))
        kc = rng.randint(min(2, len(cols)), min(max_cols, len(cols)))
        chosen = sorted(rng.sample(cols, kc))
        d = dirty.iloc[start:start + k, chosen].reset_index(drop=True)
        c = clean.iloc[start:start + k, chosen].reset_index(drop=True)
        # must contain at least one real error (the reward needs signal)
        if not any(not _cell_equal(d.iat[i, j], c.iat[i, j])
                   for i in range(len(d)) for j in range(d.shape[1])):
            continue
        buf_d, buf_c = io.StringIO(), io.StringIO()
        d.to_csv(buf_d, index=False)
        c.to_csv(buf_c, index=False)
        yield {
            "source": name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(profile_dataframe(d), d)},
            ],
            "dirty_csv": buf_d.getvalue(),
            "clean_csv": buf_c.getvalue(),
        }
        made += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/grpo_episodes.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    per = args.n // len(TRAIN_SOURCES)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for name in TRAIN_SOURCES:
            made = 0
            try:
                for ep in windows(name, rng, per):
                    f.write(json.dumps(ep, ensure_ascii=False) + "\n")
                    made += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:80]}")
            total += made
            print(f"  {name}: {made} episodes", flush=True)
    print(f"{total} episodes -> {args.out}")


if __name__ == "__main__":
    main()
