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
    """Precision/recall/F1 (convention-normalized) + damage rate.

    Churn-neutral: a rewrite that is sem-equal to the INPUT but does not restore the
    gold (pure case/whitespace churn) counts as NOTHING — not a change, not a fix, not
    damage. Otherwise bulk convention-rewrites of clean columns inflate precision (the
    '-case match' ablation artifact). Fixing an error also requires actually ACTING:
    a case-injected error left untouched is sem-equal to gold but is NOT a fix."""
    n = min(len(dirty), len(out), len(clean))
    errors = fixed = changed = good_changes = clean_cells = damage = errors_abstained = 0
    for j, col in enumerate(dirty.columns):
        present = col in out.columns
        for i in range(n):
            dv, cv = dirty.iat[i, j], clean.iat[i, j]
            ov = out.iloc[i][col] if present else dv
            err = not _cell_equal(dv, cv)                # benchmark error (raw)
            chg = present and not _cell_equal(ov, dv)    # we changed it
            raw_ok = present and _cell_equal(ov, cv)     # exactly restored gold
            sem_ok = _sem_equal(ov, cv)                  # right value (convention-tolerant)
            if chg and _sem_equal(ov, dv) and not raw_ok:
                chg = False                              # churn: ignore entirely
            if err:
                errors += 1
                if raw_ok or (sem_ok and chg):
                    fixed += 1                           # real restoration or semantic fix
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
    # traps must NOT be single-edit variants of any reference entity (else mapping them
    # is arguably CORRECT and the trap mis-scores grounding): garbage strings + one real
    # rare city (must stay; catches freq-cluster over-merge into the dominant values).
    traps = ["Xqzzyville", "Qwortelby", "Zzanthor Flats", "Carmel"]
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


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---- the validation SUITE: Raha real-error pairs + injected harvested domains ----
RAHA = [("hospital", "health"), ("beers", "beverages"), ("flights", "travel"),
        ("rayyan", "citations"), ("movies_1", "entertainment")]
# diverse subset of the 20+ harvested clean domains (cached locally)
SUITE_DOMAINS = ["restaurants", "business", "jobs", "complaints", "film", "transport",
                 "education", "music", "contractors", "alcohol-bars", "vehicles",
                 "sf-business", "la-business", "real-estate", "food-inspections"]
INJECT = {"typo": "canonicalization", "ocr": "canonicalization",
          "case": "format", "whitespace": "format"}


def _raha_pair(name):
    dirty, clean = _fetch(name)
    if len(dirty) > 2200:                       # movies_1: subsample for speed
        dirty, clean = dirty.head(2000).reset_index(drop=True), clean.head(2000).reset_index(drop=True)
    return dirty, clean


def _injected_pair(path, error_type, seed):
    import pandas as _pd
    from .inject import inject
    clean = _pd.read_csv(path, dtype=str, keep_default_na=False, nrows=600, on_bad_lines="skip")
    dirty = inject(clean, error_type, seed)
    return (dirty, clean) if dirty is not None else None


def build_suite(seed: int = 7):
    """List of specs: {name, domain, error_type, source, load() -> (dirty, clean) | None}.
    source 'real' = curated real-error benchmarks; 'injected' = seeded synthetic errors on
    harvested clean domains. Report both slices — injected typos are in-distribution for
    frequency clustering (canonical always present+dominant by construction), so the
    grounding claim lives in the REAL slice."""
    from pathlib import Path
    specs = [{"name": n, "domain": d, "error_type": "mixed", "source": "real",
              "load": (lambda n=n: _raha_pair(n))} for n, d in RAHA]
    import json
    src = {s["name"]: s["domain"] for s in
           json.load(open("training/unpaired_sources.json"))}
    cache = Path("data/real/cache")
    for fname, domain in src.items():
        if domain not in SUITE_DOMAINS:
            continue
        p = cache / f"{fname}.csv"
        if not p.exists():
            continue
        for et, group in INJECT.items():
            specs.append({"name": f"{domain}:{et}", "domain": domain, "error_type": group,
                          "source": "injected",
                          "load": (lambda p=p, et=et: _injected_pair(p, et, seed))})
    return specs


def evaluate_suite(planner, seed: int = 7) -> dict:
    """Run a planner over the whole suite at one injection seed; return the double-macro."""
    import collections
    rows = []
    for spec in build_suite(seed=seed):
        try:
            loaded = spec["load"]()
        except Exception:  # noqa: BLE001
            continue
        if loaded is None:
            continue
        dirty, clean = loaded
        cleaned, _ = apply_plan(dirty, _cell_only(planner(dirty)))
        m = score(dirty, clean, cleaned)
        rows.append({**spec, "f1": m["f1"], "damage": m["damage"]})
    by_et = collections.defaultdict(list)
    by_dom = collections.defaultdict(list)
    by_src = collections.defaultdict(list)
    for r in rows:
        by_et[r["error_type"]].append(r["f1"])
        by_dom[r["domain"]].append(r["f1"])
        by_src[r.get("source", "injected")].append(r["f1"])
    et_macro = _mean(_mean(v) for v in by_et.values())
    dom_macro = _mean(_mean(v) for v in by_dom.values())
    north = (2 * et_macro * dom_macro / (et_macro + dom_macro)) if (et_macro + dom_macro) else 0.0
    ab = abstain_slice(planner)
    return {"north": north, "et_macro": et_macro, "dom_macro": dom_macro,
            "real": _mean(by_src.get("real", [])), "injected": _mean(by_src.get("injected", [])),
            "damage": _mean(r["damage"] for r in rows), "abstain": ab["abstain_accuracy"],
            "n": len(rows), "by_et": {k: _mean(v) for k, v in by_et.items()}}


def main(seeds=(7, 17, 27), out: str | None = None) -> None:
    from scrubdata.baselines import openrefine_fingerprint_plan, openrefine_knn_plan
    systems = [("grounded (ours)", mock_plan),
               ("OpenRefine fingerprint", openrefine_fingerprint_plan),
               ("OpenRefine kNN", openrefine_knn_plan),
               ("no-op", lambda df: {"table_operations": [], "columns": [], "flags": []})]
    print("\n=== Cleaning north-star — WIDE validation suite (Raha + injected harvested), "
          f"{len(seeds)} seeds ===\n")
    print(f"{'system':<24}{'NORTH*':>9}{'±95%CI':>9}{'REAL-F1':>9}{'INJ-F1':>8}"
          f"{'damage':>8}{'abstain':>9}")
    print("-" * 76)
    table = []
    for name, planner in systems:
        norths, results = [], []
        for s in seeds:
            r = evaluate_suite(planner, seed=s)
            norths.append(r["north"]); results.append(r)
        mean_n = _mean(norths)
        # 95% CI ~ 1.96 * std / sqrt(n)
        var = _mean([(x - mean_n) ** 2 for x in norths])
        ci = 1.96 * (var ** 0.5) / (len(norths) ** 0.5)
        last = results[-1]
        row = {"system": name, "north": mean_n, "north_ci": ci,
               "real_f1": _mean(r["real"] for r in results),
               "real_f1_per_seed": [r["real"] for r in results],
               "inj_f1": _mean(r["injected"] for r in results),
               "damage": _mean(r["damage"] for r in results),
               "abstain": _mean(r["abstain"] for r in results),
               "n_datasets": last["n"], "seeds": list(seeds)}
        table.append(row)
        print(f"{name:<24}{mean_n:>9.3f}{ci:>9.3f}{row['real_f1']:>9.3f}"
              f"{row['inj_f1']:>8.3f}{row['damage']:>8.3f}{row['abstain']:>9.3f}", flush=True)
    if out:
        import json
        json.dump(table, open(out, "w"), indent=1)
        print(f"rows written to {out}")
    print(f"\nNORTH* = harmonic mean(error-type macro, domain macro) over {last['n']} datasets. "
          "Double-macro + damage + abstain = un-gameable; hospital is 1 dataset of many. "
          "The money result: grounded vs the tool people actually use (OpenRefine).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="write table rows to JSON")
    main(out=ap.parse_args().out)
