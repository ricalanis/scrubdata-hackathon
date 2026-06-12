"""W1.a — matched-supervision-budget curve: REAL-F1 vs gold labels, ours vs Baran.

OUR k-label arm (deterministic label calibration, no retraining): k gold-labeled cells
per dataset (sampled from that dataset's real error set, seed-controlled) are used ONLY
to validate/expand the planner's committed accept set — never to peek at any other gold:
  * VALIDATE: a label that refutes a committed mapping entry (raw -> wrong canon)
    overrides it with the labeled repair (raw -> gold);
  * EXPAND: a label whose surface form the planner abstained on adds the label-confirmed
    mapping (the product loop: the k labels are k resolved review flags);
  * two labels giving the SAME surface form DIFFERENT golds are contradictory evidence
    for a column-level mapping -> both dropped (abstain stays).
A mapping repairs every occurrence of the labeled surface form in its column, so one
label can fix more than its own cell. tau-lowering is a no-op on this path: the shipped
heuristic commits its full accept set at k=0 (there is no sub-tau reserve to admit).

k=0 IS the shipped pipeline (money_table.json "grounded (ours)" REAL-F1) — recomputed
here only as a pipeline-identity assertion, not a new number.

Baran points (same slice, same churn-neutral score): 20 labels = existing
eval/results/baran_raha.json; 0/5 labels = CSVs from
    uv run --no-project --python 3.10 --with "raha==1.26" --with "numpy<2" \
      --with "pandas<2.1" --with "scikit-learn<1.4" \
      python eval/run_baran.py --n-labels {0,5} --seeds 0,1 --out eval/results/baran_n{0,5}

    uv run python -m eval.label_curve                           # -> eval/results/label_curve.json
    uv run --with matplotlib python -m eval.label_curve --plot  # -> docs/paper/fig_label_curve.pdf
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .metrics import _cell_equal
from .run_real_multi import RAHA, _cell_only, _raha_pair, _sem_equal, score

RESULTS = Path(__file__).resolve().parent / "results"
PAPER = Path(__file__).resolve().parent.parent / "docs" / "paper"
KS = (0, 5, 20)
LABEL_SEEDS = (0, 1, 2)        # seed 0 = primary (spec); 1,2 = sampling variance


def sample_labels(dirty, clean, k: int, seed: int) -> list[tuple[int, str, str, str]]:
    """k gold-labeled error cells: (row, col, dirty_surface, gold). Deterministic:
    error set enumerated column-major (as score() walks it), sampled with seed."""
    n = min(len(dirty), len(clean))
    errs = [(i, j) for j in range(len(dirty.columns)) for i in range(n)
            if not _cell_equal(dirty.iat[i, j], clean.iat[i, j])]
    sel = random.Random(seed).sample(errs, min(k, len(errs)))
    return [(i, dirty.columns[j], str(dirty.iat[i, j]).strip(), str(clean.iat[i, j]))
            for i, j in sel]


def calibrate_plan(plan: dict, labels: list[tuple[int, str, str, str]]) -> dict:
    """Apply the k labels to the plan's accept set (VALIDATE/EXPAND/contradiction rules
    from the module docstring). Touches ONLY the labeled surface forms."""
    out = copy.deepcopy(plan)
    by_col: dict[str, dict[str, set]] = {}
    for _, col, raw, gold in labels:
        by_col.setdefault(col, {}).setdefault(raw, set()).add(gold)
    cols = {c.get("name"): c for c in out.setdefault("columns", [])}
    for col, repairs in by_col.items():
        repairs = {r: next(iter(g)) for r, g in repairs.items() if len(g) == 1}
        if not repairs:
            continue                                    # all contradictory -> abstain
        c = cols.get(col)
        if c is None:
            c = {"name": col, "detected_semantic_type": "categorical",
                 "issues": ["label_confirmed_errors"], "operations": []}
            out["columns"].append(c)
            cols[col] = c
        op = next((o for o in c.setdefault("operations", [])
                   if o.get("op") == "canonicalize_categories"), None)
        if op is None:
            op = {"op": "canonicalize_categories", "mapping": {},
                  "rationale": "k-label calibration: label-confirmed repairs"}
            c["operations"].append(op)
        m = op.setdefault("mapping", {})
        for raw, gold in repairs.items():
            if raw in m and _sem_equal(m[raw], gold):
                continue                                # VALIDATE: label confirms entry
            m[raw] = gold                               # override refuted / EXPAND abstained
    return out


def run_ours(ks=KS, label_seeds=LABEL_SEEDS) -> dict:
    """Label-efficiency curve for the shipped deterministic pipeline; macro REAL-F1."""
    per_ds: dict[str, dict] = {}
    points = []
    plans, pairs = {}, {}
    for name, _ in RAHA:
        pairs[name] = _raha_pair(name)
        plans[name] = mock_plan(pairs[name][0])         # deterministic; computed once
    for k in ks:
        seeds = [0] if k == 0 else list(label_seeds)
        seed_macro, seed_rows = [], []
        for s in seeds:
            rows = {}
            for name, _ in RAHA:
                dirty, clean = pairs[name]
                plan = plans[name] if k == 0 else \
                    calibrate_plan(plans[name], sample_labels(dirty, clean, k, s))
                cleaned, _ = apply_plan(dirty, _cell_only(plan))
                m = score(dirty, clean, cleaned)
                rows[name] = {kk: m[kk] for kk in ("f1", "precision", "recall", "damage")}
            macro = sum(r["f1"] for r in rows.values()) / len(rows)
            seed_macro.append(macro)
            seed_rows.append(rows)
            detail = ", ".join(f"{n} {r['f1']:.3f}" for n, r in rows.items())
            print(f"  ours k={k} seed={s}: macro REAL-F1 {macro:.3f} ({detail})", flush=True)
        mu = sum(seed_macro) / len(seed_macro)
        sd = (sum((x - mu) ** 2 for x in seed_macro) / len(seed_macro)) ** 0.5
        points.append({"k": k, "macro_f1": mu, "macro_f1_sd": sd,
                       "macro_f1_per_seed": seed_macro, "label_seeds": seeds,
                       "per_dataset_seed0": seed_rows[0]})
        if k == 0:
            per_ds = seed_rows[0]
    shipped = json.load(open(RESULTS / "money_table.json"))[0]["real_f1"]
    drift = abs(points[0]["macro_f1"] - shipped)
    if drift > 1e-9:                       # planner code moved since money_table.json
        print(f"  NOTE: k=0 at HEAD {points[0]['macro_f1']:.4f} != money_table "
              f"{shipped:.4f} — money_table.json is stale (re-run needed for paper)")
    return {"system": "ScrubData (grounded heuristic + k-label calibration)",
            "label_use": "validate/expand accept set only; no retraining, no extra gold",
            "points": points, "shipped_k0_money_table": shipped,
            "k0_drift_vs_money_table": drift, "per_dataset_k0": per_ds}


def baran_point(repaired_dir: Path, n_labels: int) -> dict:
    """Score one Baran budget from its repaired CSVs under the identical protocol."""
    csvs = sorted(repaired_dir.glob("*_seed*_repaired.csv")) if repaired_dir.exists() else []
    if not csvs:
        return {"k": n_labels, "status": "cannot_run",
                "reason": f"no repaired CSVs in {repaired_dir} (run crashed or not run)"}
    import collections

    import pandas as pd
    per_seed = collections.defaultdict(dict)
    for p in csvs:
        name, seed = p.stem.rsplit("_repaired", 1)[0].rsplit("_seed", 1)
        dirty, clean = _raha_pair(name)
        out = pd.read_csv(p, dtype=str, keep_default_na=False)
        m = score(dirty, clean, out)
        per_seed[int(seed)][name] = {kk: m[kk] for kk in ("f1", "precision", "recall", "damage")}
    seed_macro = [sum(r["f1"] for r in rows.values()) / len(rows)
                  for rows in per_seed.values()]
    mu = sum(seed_macro) / len(seed_macro)
    sd = (sum((x - mu) ** 2 for x in seed_macro) / len(seed_macro)) ** 0.5
    return {"k": n_labels, "status": "ok", "macro_f1": mu, "macro_f1_sd": sd,
            "macro_f1_per_seed": seed_macro, "n_seeds": len(per_seed),
            "per_dataset_seed0": per_seed.get(0, {})}


def assemble(out: Path) -> dict:
    print("ScrubData arm:", flush=True)
    ours = run_ours()
    baran20 = json.load(open(RESULTS / "baran_raha.json"))
    baran = {"system": "Baran (oracle error positions + k gold-labeled tuples)",
             "protocol_note": baran20["protocol_note"],
             "points": [baran_point(RESULTS / "baran_n0", 0),
                        baran_point(RESULTS / "baran_n5", 5),
                        {"k": 20, "status": "ok", "macro_f1": baran20["real_f1"],
                         "macro_f1_sd": None, "ci95": baran20["real_f1_ci"],
                         "macro_f1_per_seed": baran20["real_f1_per_seed"],
                         "n_seeds": baran20["n_seeds"],
                         "source": "eval/results/baran_raha.json (existing 3-seed run)"}]}
    result = {
        "metric": "churn-neutral REAL-F1 (eval.run_real_multi.score), "
                  "macro over the 5 Raha real-error datasets",
        "datasets": [n for n, _ in RAHA],
        "ours": ours, "baran": baran,
        "honesty": "k-label arm touches only the k sampled (cell, gold) pairs per "
                   "dataset; Baran additionally receives oracle error positions at "
                   "every k (disclosed asymmetry in its favor).",
    }
    json.dump(result, open(out, "w"), indent=1)
    print(f"-> {out}")
    return result


def plot(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    o = result["ours"]["points"]
    ax.errorbar([p["k"] for p in o], [p["macro_f1"] for p in o],
                yerr=[p["macro_f1_sd"] for p in o], marker="o", ms=5, lw=1.6,
                color="#2f6f5e", capsize=3, label="ScrubData (zero-config; labels only\ncalibrate the accept set)")
    bp = [p for p in result["baran"]["points"] if p.get("status") == "ok"]
    bx, by = [p["k"] for p in bp], [p["macro_f1"] for p in bp]
    ax.errorbar(bx, by, yerr=[p.get("macro_f1_sd") or 0 for p in bp], marker="s", ms=5,
                lw=1.6, ls="--", color="#b3433b", capsize=3,
                label="Baran (oracle error positions\n+ k labeled tuples)")
    b0 = result["baran"]["points"][0]
    if b0.get("status") == "cannot_run":
        ax.scatter([0], [0], marker="x", s=60, color="#b3433b", zorder=5)
        ax.annotate("Baran cannot run\nwithout labels", xy=(0, 0), xytext=(1.2, 0.10),
                    fontsize=8, color="#b3433b",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#b3433b"))
    elif b0.get("macro_f1", 1) < 0.01:
        ax.annotate("at 0 labels Baran runs but\nrepairs nothing (F1 0.000)",
                    xy=(0, b0["macro_f1"]), xytext=(2.6, 0.05), fontsize=8,
                    color="#b3433b",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#b3433b"))
    ax.set_xlabel("gold-labeled cells per dataset (supervision budget)")
    ax.set_ylabel("REAL-F1 (macro, 5 Raha datasets)")
    ax.set_xticks([0, 5, 20])
    ax.set_ylim(-0.04, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="render docs/paper/fig_label_curve.pdf from the JSON")
    ap.add_argument("--out", default=str(RESULTS / "label_curve.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if args.plot:
        plot(json.load(open(out)), PAPER / "fig_label_curve.pdf")
    else:
        assemble(out)


if __name__ == "__main__":
    main()
