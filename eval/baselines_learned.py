"""WS4 learned-repair baselines: scoring + Jellyfish prompt construction.

Both baselines bypass plan dicts (the executor is column-level by design; learned repair
is per-cell) — they produce repaired DataFrames scored by the SAME churn-neutral
`eval.run_real_multi.score` as every other row of the money table.

* Baran: repaired CSVs come from eval/run_baran.py (pinned env). Score here:
      uv run python -m eval.baselines_learned --score-baran
* Jellyfish: prompts built here (unit-testable without a GPU), executed by
  scripts/modal_jellyfish.py (vLLM on Modal), scored in-run with the same `score`.

Jellyfish has NO repair task — we compose its two published cell-level tasks:
error detection (yes/no per cell) then data imputation (infer the flagged cell with the
attribute removed). Prompt templates are verbatim from the NECOUDBFM/Jellyfish-13B model
card; this composition is OURS, not theirs (disclosed in the paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_MESSAGE = ("You are an AI assistant that follows instruction extremely well. "
                  "Help as much as you can.")

_ED_TEMPLATE = (
    "Your task is to determine if there is an error in the value of a specific "
    "attribute within the whole record provided.\n"
    "The attributes may include {attrs}.\n"
    "Errors may include, but are not limited to, spelling errors, inconsistencies, "
    "or values that don't make sense given the context of the whole record.\n"
    "Record [{record}]\n"
    "Attribute for Verification: [{col}: {val}]\n"
    "Question: Is there an error in the value of {col}? "
    "Choose your answer from: [Yes, No]."
)

_DI_TEMPLATE = (
    "You are presented with a {keyword} record that is missing a specific attribute: "
    "{col}.\n"
    "Your task is to deduce or infer the value of {col} using the available "
    "information in the record.\n"
    "You may be provided with fields like {attrs} to help you in the inference.\n"
    "Record: [{record}]\n"
    "Based on the provided record, what would you infer is the value for the missing "
    "attribute {col}?\n"
    "Answer only the value of {col}."
)


def wrap_prompt(user_message: str) -> str:
    """The Jellyfish-13B chat scaffold (verbatim from the model card)."""
    return f"{SYSTEM_MESSAGE}\n\n### Instruction:\n\n{user_message}\n\n### Response:\n\n"


def _serialize(record: dict, skip: str | None = None) -> str:
    return ", ".join(f"{k}: {v}" for k, v in record.items() if k != skip)


def ed_prompt(record: dict, col: str) -> str:
    """Error-detection prompt (whole-record form) for one cell."""
    return wrap_prompt(_ED_TEMPLATE.format(
        attrs=", ".join(record.keys()), record=_serialize(record),
        col=col, val=record[col]))


def di_prompt(record: dict, col: str, keyword: str) -> str:
    """Data-imputation prompt for a flagged cell — the attribute is REMOVED from the
    serialized record so the model infers, not copies."""
    attrs = [k for k in record.keys() if k != col]
    return wrap_prompt(_DI_TEMPLATE.format(
        keyword=keyword, col=col, attrs=", ".join(attrs),
        record=_serialize(record, skip=col)))


def parse_ed(text: str) -> bool:
    """True = the model says the cell is erroneous."""
    return text.strip().lower().lstrip("[").startswith("yes")


def parse_di(text: str, original: str) -> str:
    """Imputed value, or the original (abstain) when the answer is unusable —
    empty, multi-line/rambling, or implausibly long for a cell."""
    ans = text.strip().strip('"').strip()
    if not ans or "\n" in ans or len(ans) > 80:
        return original
    return ans


# ---------------------------------------------------------------- Baran scoring

def score_baran(repaired_dir: str = "eval/results/baran",
                out: str = "eval/results/baran_raha.json") -> dict:
    """Score every <name>_seed<k>_repaired.csv against (dirty, clean) under the
    identical churn-neutral protocol; macro REAL-F1 mean ± 95% CI over seeds."""
    import collections

    import pandas as pd

    from .run_real_multi import _raha_pair, score

    per_seed: dict[int, list] = collections.defaultdict(list)
    per_ds = []
    for p in sorted(Path(repaired_dir).glob("*_seed*_repaired.csv")):
        name, seed = p.stem.rsplit("_repaired", 1)[0].rsplit("_seed", 1)
        repaired = pd.read_csv(p, dtype=str, keep_default_na=False)
        dirty, clean = _raha_pair(name)
        m = score(dirty, clean, repaired)
        per_seed[int(seed)].append(m)
        per_ds.append({"name": name, "seed": int(seed), **{k: v for k, v in m.items()}})
        print(f"  {name:<10} seed{seed}: F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} dmg={m['damage']:.3f}")
    if not per_seed:
        raise SystemExit(f"no repaired CSVs found in {repaired_dir}")

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    seed_f1 = [mean(m["f1"] for m in ms) for ms in per_seed.values()]
    mu = mean(seed_f1)
    var = mean([(x - mu) ** 2 for x in seed_f1])
    ci = 1.96 * (var ** 0.5) / (len(seed_f1) ** 0.5)
    result = {
        "system": "Baran (oracle detection, 20 gold labels)",
        "real_f1": mu, "real_f1_ci": ci, "real_f1_per_seed": seed_f1,
        "damage": mean(mean(m["damage"] for m in ms) for ms in per_seed.values()),
        "precision": mean(mean(m["precision"] for m in ms) for ms in per_seed.values()),
        "recall": mean(mean(m["recall"] for m in ms) for ms in per_seed.values()),
        "n_seeds": len(per_seed), "per_dataset": per_ds,
        "protocol_note": "upper bound: oracle error positions + 20 gold-labeled tuples "
                         "(its package default); damage=0 by construction",
    }
    json.dump(result, open(out, "w"), indent=1)
    print(f"\nBaran macro REAL-F1 {mu:.3f} ± {ci:.3f} (n={len(seed_f1)} seeds) -> {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-baran", action="store_true")
    args = ap.parse_args()
    if args.score_baran:
        score_baran()
