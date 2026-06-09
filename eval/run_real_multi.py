"""Improved real-data north-star — fixes the biases of single-dataset `repair_recall`.

repair_recall on hospital alone is biased: (1) ONE dataset (overfit to its quirks),
(2) RECALL-ONLY so it rewards over-correction (v5: recall .42 / precision .16),
(3) CONVENTION-sensitive (penalizes case/format differences that are semantically right),
(4) ABSTAIN-blind (punishes correctly declining on uncertain values).

This harness fixes all four:
  * MULTI-DATASET (Raha suite) -> macro-average, not one table.
  * F1 (recall AND precision) + DAMAGE rate (clean cells we corrupted) -> over-correction
    is penalized, not rewarded.
  * CONVENTION-NORMALIZED correctness (case/whitespace-insensitive) -> measures the right
    VALUE, not surface convention.
  * an ADVERSARIAL ABSTAIN slice -> the system must fix real typos but LEAVE trap values
    (ambiguous / not-a-typo) untouched. Abstaining correctly scores; wrong-merging doesn't.

    uv run python -m eval.run_real_multi
"""

from __future__ import annotations

import random

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .metrics import _cell_equal

DATASETS = ["hospital", "beers", "flights", "rayyan"]
RAW = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets"


def _cell_only(plan: dict) -> dict:
    """Strip row-count-changing table ops (dedup / drop-empty-rows) so the cleaned output
    stays row-aligned with the reference — we are measuring CELL-level cleaning here."""
    p = dict(plan)
    p["table_operations"] = [o for o in plan.get("table_operations", [])
                             if o.get("op") not in ("drop_exact_duplicates", "drop_empty_rows")]
    return p


def _sem_equal(a, b) -> bool:
    """Semantic equality: numeric-tolerant (via _cell_equal) OR case/whitespace-folded.
    Separates the right VALUE from surface convention (Birmingham == birmingham)."""
    if _cell_equal(a, b):
        return True
    return str(a).strip().lower() == str(b).strip().lower()


def _fetch(name: str):
    import urllib.request
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent / "data" / "real" / name
    base.mkdir(parents=True, exist_ok=True)
    out = []
    for fn in ("dirty.csv", "clean.csv"):
        p = base / fn
        if not p.exists():
            urllib.request.urlretrieve(f"{RAW}/{name}/{fn}", p)
        out.append(pd.read_csv(p, dtype=str, keep_default_na=False))
    return out


def score(dirty: pd.DataFrame, clean: pd.DataFrame, out: pd.DataFrame) -> dict:
    """Precision/recall/F1 (convention-normalized) + damage rate + abstain count."""
    n = min(len(dirty), len(out), len(clean))
    errors = fixed = changed = good_changes = clean_cells = damage = errors_abstained = 0
    for j, col in enumerate(dirty.columns):
        present = col in out.columns
        for i in range(n):
            dv, cv = dirty.iat[i, j], clean.iat[i, j]
            ov = out.iloc[i][col] if present else dv
            err = not _cell_equal(dv, cv)               # benchmark error (raw)
            chg = present and not _cell_equal(ov, dv)    # we changed it
            sem_ok = _sem_equal(ov, cv)                  # got the right value (semantic)
            if err:
                errors += 1
                if sem_ok:
                    fixed += 1
                elif not chg:
                    errors_abstained += 1                # left an error untouched
            else:
                clean_cells += 1
                if chg and not sem_ok:
                    damage += 1                          # corrupted a clean cell
            if chg:
                changed += 1
                if sem_ok:
                    good_changes += 1
    recall = fixed / errors if errors else 0.0
    precision = good_changes / changed if changed else 1.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {"f1": f1, "recall": recall, "precision": precision,
            "damage": damage / clean_cells if clean_cells else 0.0,
            "_errors": errors, "_changed": changed, "_fixed": fixed}


def abstain_slice(planner, seed: int = 3) -> dict:
    """Adversarial: a city column with real typos (must fix) + TRAP values (must leave).
    Tests that the system abstains rather than wrong-merges."""
    rng = random.Random(seed)
    canon = ["Chicago", "Boston", "Houston", "Phoenix", "Dallas"]
    col, gold, kind = [], [], []
    for c in canon:                                      # clean anchors
        for _ in range(8):
            col.append(c); gold.append(c); kind.append("clean")
    typos = {"Chcago": "Chicago", "Bostton": "Boston", "Houston": "Houston"}
    for t, g in typos.items():                           # real typos -> must FIX
        col.append(t); gold.append(g); kind.append("typo")
    traps = ["Boazz", "Sprngfield", "Xqzzyville", "Carmel"]  # not-a-present-typo -> must LEAVE
    for t in traps:
        col.append(t); gold.append(t); kind.append("trap")
    idx = list(range(len(col))); rng.shuffle(idx)
    col = [col[i] for i in idx]; gold = [gold[i] for i in idx]; kind = [kind[i] for i in idx]
    df = pd.DataFrame({"city": col})
    cleaned, _ = apply_plan(df, _cell_only(planner(df)))
    out = cleaned["city"].tolist() if "city" in cleaned.columns else col
    typo_fixed = sum(1 for o, g, k in zip(out, gold, kind) if k == "typo" and _sem_equal(o, g))
    trap_left = sum(1 for o, g, k in zip(out, gold, kind) if k == "trap" and _sem_equal(o, g))
    n_typo = kind.count("typo"); n_trap = kind.count("trap")
    return {"typo_recall": typo_fixed / n_typo, "abstain_accuracy": trap_left / n_trap,
            "_typos": n_typo, "_traps": n_trap}


def main() -> None:
    print("\n=== Improved real-data north-star (multi-dataset, precision-aware, "
          "convention-normalized) ===\n")
    hdr = f"{'dataset':<10}{'F1':>8}{'recall':>8}{'precision':>11}{'damage':>9}{'errors':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    f1s, dmgs = [], []
    for name in DATASETS:
        try:
            dirty, clean = _fetch(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<10} fetch failed ({type(e).__name__})")
            continue
        cleaned, _ = apply_plan(dirty, _cell_only(mock_plan(dirty)))
        m = score(dirty, clean, cleaned)
        f1s.append(m["f1"]); dmgs.append(m["damage"])
        print(f"{name:<10}{m['f1']:>8.3f}{m['recall']:>8.3f}{m['precision']:>11.3f}"
              f"{m['damage']:>9.3f}{m['_errors']:>8}")
    if f1s:
        print("-" * len(hdr))
        print(f"{'MACRO':<10}{sum(f1s) / len(f1s):>8.3f}{'':>8}{'':>11}"
              f"{sum(dmgs) / len(dmgs):>9.3f}")
    ab = abstain_slice(mock_plan)
    print(f"\nADVERSARIAL abstain slice: typo_recall={ab['typo_recall']:.3f} "
          f"({ab['_typos']} typos), abstain_accuracy={ab['abstain_accuracy']:.3f} "
          f"({ab['_traps']} traps must be LEFT untouched)")
    print("\nNORTH-STAR = macro-F1 (balances fix-rate vs over-correction) + low damage + "
          "high abstain_accuracy. This can't be gamed by changing everything.")


if __name__ == "__main__":
    main()
