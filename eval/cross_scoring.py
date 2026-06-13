"""B1 (W4.2) dual-metric cross-scoring on the 5 Raha real-error datasets.

Scores every system under BOTH metric families, side by side:
  * original  — the Raha/Baran cell-level repair protocol (Mahdavi & Abedjan,
    PVLDB 13(12), p1948, Sec 6.1 + raha/dataset.py get_data_cleaning_evaluation):
    values minimally normalized (html-unescape, whitespace collapse — their
    value_normalizer), then RAW string equality; precision = exact-gold repairs /
    cells changed; recall = exact-gold repairs / (dirty->clean diff); no
    churn-neutrality, no case folding, no semantic tolerance, no damage metric.
  * churn_neutral — our eval.run_real_multi.score (the scoring contract):
    convention-normalized, churn ignored, damage reported.

Systems: grounded (HEAD mock_plan), verified union (v6, tau=0.5 — identical plan
files to eval.raha_table), OpenRefine fingerprint/kNN, and Baran at labeling
budgets 0/5/20 (oracle detection; repaired CSVs from eval/run_baran.py, 3 seeds,
seed-mean). Baran-from-CSV caveat: corrections equal to the dirty value vanish
from the repaired-vs-dirty diff, so reconstructed |changed| is a lower bound on
Baran's own output_size (precision an upper bound; recall exact).

Also computes Kendall tau-b between the SYSTEM RANKINGS induced by the two F1s
(per dataset + macro), and a calibration block: our Baran oracle+20 repro vs the
published Table 3 "Baran" row (verified from the PVLDB PDF; see PUBLISHED below).

Acceptance: the churn-neutral rows must reproduce eval/results/raha_per_dataset.json
exactly (checked, hard-fails otherwise).

    uv run python -m eval.cross_scoring
Writes eval/results/cross_scoring.json and prints LaTeX rows.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from scrubdata.baselines import openrefine_fingerprint_plan, openrefine_knn_plan
from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan
from scrubdata.verifier import union_plans, verify_plan

from .precision_curve import _repairs_only
from .raha_table import TAU, UNION_PLANS, _gen_plan
from .run_real_multi import RAHA, _cell_only, _raha_pair, score

RESULTS = Path(__file__).resolve().parent / "results"
BARAN_DIRS = {0: RESULTS / "baran_n0", 5: RESULTS / "baran_n5", 20: RESULTS / "baran"}

# Baran PVLDB'20 Table 3, row "Baran" (no TL): complete set of data errors given as
# input (= oracle detection), labeling budget 20, mean of 10 runs. Verified by reading
# vldb.org/pvldb/vol13/p1948-mahdavi.pdf p1957 (2026-06-12). movies_1 is not evaluated
# in the paper (its real-error sets are hospital/flights/address/beers/rayyan/it/tax).
PUBLISHED = {"hospital": {"precision": 0.88, "recall": 0.86, "f1": 0.87},
             "flights": {"precision": 1.00, "recall": 1.00, "f1": 1.00},
             "beers": {"precision": 0.91, "recall": 0.89, "f1": 0.90},
             "rayyan": {"precision": 0.76, "recall": 0.40, "f1": 0.52}}


def _norm(v: str) -> str:
    """raha.dataset.Dataset.value_normalizer, verbatim semantics."""
    v = html.unescape(str(v))
    v = re.sub("[\t\n ]+", " ", v, re.UNICODE)
    return v.strip("\t\n ")


def baran_score(dirty: pd.DataFrame, clean: pd.DataFrame, out: pd.DataFrame) -> dict:
    """The original Raha/Baran repair metric over a repaired DataFrame: minimal
    normalization then raw equality; changed = repaired-vs-dirty diff."""
    n = min(len(dirty), len(out), len(clean))
    errors = changed = tp = 0
    for j, col in enumerate(dirty.columns):
        present = col in out.columns
        for i in range(n):
            dv, cv = _norm(dirty.iat[i, j]), _norm(clean.iat[i, j])
            ov = _norm(out.iloc[i][col]) if present else dv
            err, chg = dv != cv, ov != dv
            errors += err
            changed += chg
            tp += chg and err and ov == cv
    p = tp / changed if changed else 0.0
    r = tp / errors if errors else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": f1, "precision": p, "recall": r,
            "_errors": errors, "_changed": changed, "_tp": tp}


def _both(dirty, clean, out) -> dict:
    m = score(dirty, clean, out)
    return {"original": baran_score(dirty, clean, out),
            "churn_neutral": {k: m[k] for k in
                              ("f1", "precision", "recall", "damage",
                               "_errors", "_changed", "_fixed")}}


def kendall_tau(xs, ys) -> float:
    """Kendall tau-b (tie-corrected), stdlib."""
    n = len(xs)
    n0, n1, n2, nc, nd = n * (n - 1) // 2, 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            n1 += a == 0
            n2 += b == 0
            if a != 0 and b != 0:
                nc += (a > 0) == (b > 0)
                nd += (a > 0) != (b > 0)
    denom = ((n0 - n1) * (n0 - n2)) ** 0.5
    return (nc - nd) / denom if denom else 0.0


def _mean_rows(rows: list[dict]) -> dict:
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def main() -> None:
    out = {"protocol": {
        "original": "Raha/Baran convention: value_normalizer (html-unescape + "
                    "whitespace collapse) then raw string equality; P = exact-gold "
                    "repairs / changed cells, R = exact-gold repairs / (dirty->clean "
                    "diff); no churn-neutrality, no damage",
        "churn_neutral": "eval.run_real_multi.score — the scoring contract",
        "baran_rows": "oracle error positions + n gold labels, 3 seeds, seed-mean; "
                      "reconstructed from repaired CSVs (no-op corrections vanish: "
                      "|changed| lower-bounds Baran's output_size)",
        "movies_1": "first 2000 rows (_raha_pair), as everywhere in the suite"},
        "systems": {}}

    deterministic = [("grounded", mock_plan),
                     ("openrefine_fingerprint", openrefine_fingerprint_plan),
                     ("openrefine_knn", openrefine_knn_plan)]
    for label, planner in deterministic:
        rows = []
        for name, _dom in RAHA:
            dirty, clean = _raha_pair(name)
            cleaned, _ = apply_plan(dirty, _cell_only(planner(dirty)))
            m = _both(dirty, clean, cleaned)
            rows.append({"dataset": name, **m})
            print(f"  {label:<24}{name:<10} orig={m['original']['f1']:.3f} "
                  f"cn={m['churn_neutral']['f1']:.3f}", flush=True)
        out["systems"][label] = {"per_dataset": rows}

    rows = []
    for name, _dom in RAHA:
        base = (json.load(open(UNION_PLANS[name])) if name in UNION_PLANS
                else _gen_plan(name))
        dirty, clean = _raha_pair(name)
        plan = _repairs_only(union_plans(verify_plan(dirty, base, tau=TAU),
                                         mock_plan(dirty)))
        cleaned, _ = apply_plan(dirty, plan)
        m = _both(dirty, clean, cleaned)
        rows.append({"dataset": name, **m})
        print(f"  {'verified_union':<24}{name:<10} orig={m['original']['f1']:.3f} "
              f"cn={m['churn_neutral']['f1']:.3f}", flush=True)
    out["systems"]["verified_union_v6_tau0.5"] = {"per_dataset": rows}

    for n_labels, d in BARAN_DIRS.items():
        rows = []
        for name, _dom in RAHA:
            dirty, clean = _raha_pair(name)
            per_seed = []
            for p in sorted(d.glob(f"{name}_seed*_repaired.csv")):
                repaired = pd.read_csv(p, dtype=str, keep_default_na=False)
                per_seed.append(_both(dirty, clean, repaired))
            m = {"original": _mean_rows([s["original"] for s in per_seed]),
                 "churn_neutral": _mean_rows([s["churn_neutral"] for s in per_seed])}
            rows.append({"dataset": name, "n_seeds": len(per_seed), **m})
            print(f"  {'baran_oracle%d' % n_labels:<24}{name:<10} "
                  f"orig={m['original']['f1']:.3f} "
                  f"cn={m['churn_neutral']['f1']:.3f}", flush=True)
        out["systems"][f"baran_oracle{n_labels}"] = {"per_dataset": rows}

    for sys in out["systems"].values():
        for fam in ("original", "churn_neutral"):
            sys[f"macro_f1_{fam}"] = _mean_rows(
                [r[fam] for r in sys["per_dataset"]])["f1"]

    # acceptance: churn-neutral rows == raha_per_dataset.json (exact)
    ref = json.load(open(RESULTS / "raha_per_dataset.json"))
    checks = []
    for key, ref_key in [("grounded", "grounded"),
                         ("openrefine_fingerprint", "openrefine_fingerprint"),
                         ("openrefine_knn", "openrefine_knn"),
                         ("verified_union_v6_tau0.5", "verified_union_v6_tau0.5"),
                         ("baran_oracle20", "baran_oracle20")]:
        for got, want in zip(out["systems"][key]["per_dataset"],
                             ref["systems"][ref_key]["per_dataset"]):
            for k in ("f1", "precision", "recall", "damage"):
                ok = abs(got["churn_neutral"][k] - want[k]) < 1e-9
                checks.append(ok)
                if not ok:
                    print(f"MISMATCH {key}/{got['dataset']}/{k}: "
                          f"{got['churn_neutral'][k]} vs {want[k]}")
    out["acceptance"] = {"vs": "raha_per_dataset.json", "n_cells": len(checks),
                         "pass": all(checks)}
    print(f"\nacceptance: {sum(checks)}/{len(checks)} cells match "
          f"-> {'PASS' if all(checks) else 'FAIL'}")
    if not all(checks):
        raise SystemExit("acceptance FAILED")

    # Kendall tau-b between system rankings under the two F1s
    primary = ["grounded", "verified_union_v6_tau0.5", "openrefine_fingerprint",
               "openrefine_knn", "baran_oracle20"]
    extended = primary + ["baran_oracle0", "baran_oracle5"]
    taus = {}
    for label, sysset in [("primary", primary), ("extended", extended)]:
        per_ds = {}
        for i, (name, _dom) in enumerate(RAHA):
            xs = [out["systems"][s]["per_dataset"][i]["original"]["f1"] for s in sysset]
            ys = [out["systems"][s]["per_dataset"][i]["churn_neutral"]["f1"] for s in sysset]
            per_ds[name] = kendall_tau(xs, ys)
        xs = [out["systems"][s]["macro_f1_original"] for s in sysset]
        ys = [out["systems"][s]["macro_f1_churn_neutral"] for s in sysset]
        taus[label] = {"systems": sysset, "per_dataset": per_ds,
                       "macro": kendall_tau(xs, ys)}
        print(f"tau-b ({label}): macro={taus[label]['macro']:.3f}  " +
              "  ".join(f"{n}={t:.3f}" for n, t in per_ds.items()))
    out["kendall_tau_b"] = taus

    # calibration: our Baran oracle+20 repro (ORIGINAL metric) vs published Table 3
    cal = []
    b20 = {r["dataset"]: r for r in out["systems"]["baran_oracle20"]["per_dataset"]}
    for name, pub in PUBLISHED.items():
        ours = b20[name]["original"]
        cal.append({"dataset": name, "published_f1": pub["f1"],
                    "published_precision": pub["precision"],
                    "published_recall": pub["recall"],
                    "repro_f1": ours["f1"], "repro_precision": ours["precision"],
                    "repro_recall": ours["recall"],
                    "delta_f1": ours["f1"] - pub["f1"]})
        print(f"calibration {name:<10} published F1={pub['f1']:.2f} "
              f"repro F1={ours['f1']:.3f} (d={ours['f1'] - pub['f1']:+.3f})")
    out["calibration"] = {
        "source": "Mahdavi & Abedjan, PVLDB 13(12) p1948, Table 3 row 'Baran' "
                  "(no TL): complete error set given (oracle detection), budget 20, "
                  "mean of 10 runs; PDF read 2026-06-12",
        "notes": "their runs: full datasets, 10 label seeds, Wikipedia value models "
                 "available in package but Table-3 row is without TL; ours: 3 label "
                 "seeds, no pretraining, movies_1 not in their paper; our "
                 "churn-neutral macro for this row is the paper's 0.811",
        "rows": cal}

    dest = RESULTS / "cross_scoring.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"written to {dest}")
    print(latex(out))


LABELS = [("grounded", "Grounded (ours, deterministic)"),
          ("verified_union_v6_tau0.5", r"Verified union (v6, $\tau{=}0.5$)"),
          ("openrefine_fingerprint", "OpenRefine fingerprint"),
          ("openrefine_knn", "OpenRefine kNN"),
          ("baran_oracle20", r"Baran (oracle det.\ + 20 labels)")]


def latex(out: dict) -> str:
    """Booktabs rows: per system x dataset, original P/R/F1 next to churn-neutral
    P/R/F1 + damage."""
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{Original (Baran) metric} & "
         r"\multicolumn{4}{c}{Churn-neutral (ours)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
         r"System & Dataset & Prec. & Rec. & F1 & Prec. & Rec. & F1 & Damage \\",
         r"\midrule"]
    for key, label in LABELS:
        for i, r in enumerate(out["systems"][key]["per_dataset"]):
            o, c = r["original"], r["churn_neutral"]
            L.append(f"{label if i == 0 else ''} & "
                     f"{r['dataset'].replace('_', r'\_')} & "
                     f"{o['precision']:.3f} & {o['recall']:.3f} & {o['f1']:.3f} & "
                     f"{c['precision']:.3f} & {c['recall']:.3f} & {c['f1']:.3f} & "
                     f"{c['damage']:.3f} \\\\")
        L.append(f" & \\emph{{macro}} &  &  & "
                 f"\\emph{{{out['systems'][key]['macro_f1_original']:.3f}}} &  &  & "
                 f"\\emph{{{out['systems'][key]['macro_f1_churn_neutral']:.3f}}} &  \\\\")
        L.append(r"\midrule")
    t = out["kendall_tau_b"]["primary"]
    L.append(r"\multicolumn{9}{l}{Kendall $\tau_b$ between system rankings: "
             f"macro {t['macro']:.2f}; per dataset " +
             ", ".join(f"{n.replace('_', r'\_')} {v:.2f}"
                       for n, v in t["per_dataset"].items()) + r"} \\")
    cal = ", ".join(f"{r['dataset'].replace('_', r'\_')} {r['repro_f1']:.3f} vs "
                    f"{r['published_f1']:.2f}" for r in out["calibration"]["rows"])
    L.append(r"\multicolumn{9}{l}{Calibration, original metric (our Baran oracle+20 "
             r"repro vs PVLDB'20 Table~3): " + cal + r"} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
