"""Supervisor REQUIRED 5 — per-dataset Raha table (real-error slice).

Per-dataset repair precision/recall/F1 + damage on the 5 Raha real-error benchmarks
(hospital, beers, flights, rayyan, movies_1) for:
  * grounded (ours)        — shipped deterministic path (mock_plan), money-table protocol
  * OpenRefine fingerprint — clustering baseline, same protocol
  * OpenRefine kNN         — clustering baseline, same protocol
  * verified union (v6)    — captured raw model plan -> verify(tau=0.5) -> union with
                             heuristic -> repairs-only (the WS1 gate protocol), where a
                             captured plan exists (hospital + the gen_plans_seed21 set)
  * Baran                  — reference row from eval/results/baran_raha.json (oracle
                             detection + 20 gold labels; mean over its 3 label seeds)

Deterministic (no injection): the real slice is seed-free, so the grounded macro F1
must recompute exactly to money_table.json's real_f1 — the acceptance check. (The
supervisor's 0.174 is the value at the money-table commit 536cbfb; planner commits
since then moved the HEAD macro — money_table.json is re-run in lockstep.)

    uv run python -m eval.raha_table
Writes eval/results/raha_per_dataset.json and prints LaTeX rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from scrubdata.baselines import openrefine_fingerprint_plan, openrefine_knn_plan
from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan
from scrubdata.verifier import union_plans, verify_plan

from .precision_curve import _repairs_only
from .run_real_multi import RAHA, _cell_only, _raha_pair, score

RESULTS = Path(__file__).resolve().parent / "results"
TAU = 0.5  # the pre-registered WS1 operating point

# captured raw model plans (shipped v6 = mixA seed 21). Modal bf16 captures first;
# local Q8_0 captures (eval/capture_plan_local.py) as fallback, suffix-disclosed.
UNION_PLANS = {"hospital": RESULTS / "v6_hospital_raw_plan.json"}
for _n in ("beers", "movies_1", "flights", "rayyan"):
    for suffix in ("raw_plan", "raw_plan_localq8"):
        p = RESULTS / f"v6_{_n}_{suffix}.json"
        if p.exists():
            UNION_PLANS[_n] = p
            break
_GEN = RESULTS / "gen_plans_seed21.json"  # v6 champion captures (flights, rayyan, ...)


def _gen_plan(name):
    plans = json.load(open(_GEN))
    return plans.get(name)


def _row(name, m):
    return {"dataset": name, "f1": m["f1"], "precision": m["precision"],
            "recall": m["recall"], "damage": m["damage"],
            "errors": m["_errors"], "changed": m["_changed"], "fixed": m["_fixed"]}


def main() -> None:
    table = {"protocol": {
        "deterministic_rows": "full plan minus row-dropping ops (_cell_only), "
                              "churn-neutral score — identical to run_real_multi",
        "union_rows": f"verify(tau={TAU}) -> union with heuristic -> repairs-only "
                      "(canonicalize decisions; the WS1 gate protocol)",
        "baran_row": "eval/results/baran_raha.json — oracle error positions + 20 gold "
                     "labels (upper bound), mean over 3 label-sampling seeds",
        "movies_1": "scored on first 2000 rows (_raha_pair), as in the money table"},
        "systems": {}}

    systems = [("grounded", mock_plan),
               ("openrefine_fingerprint", openrefine_fingerprint_plan),
               ("openrefine_knn", openrefine_knn_plan)]
    for label, planner in systems:
        rows = []
        for name, _dom in RAHA:
            dirty, clean = _raha_pair(name)
            cleaned, _ = apply_plan(dirty, _cell_only(planner(dirty)))
            m = score(dirty, clean, cleaned)
            rows.append(_row(name, m))
            print(f"  {label:<24}{name:<12} F1={m['f1']:.3f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} dmg={m['damage']:.4f}", flush=True)
        macro = sum(r["f1"] for r in rows) / len(rows)
        table["systems"][label] = {"per_dataset": rows, "macro_f1": macro}

    # verified-union operating point per dataset (where a raw v6 plan was captured)
    rows = []
    for name, _dom in RAHA:
        if name in UNION_PLANS:
            base, src = json.load(open(UNION_PLANS[name])), UNION_PLANS[name].name
        else:
            base, src = _gen_plan(name), _GEN.name
        if base is None:
            rows.append({"dataset": name, "missing": "no captured v6 raw plan"})
            print(f"  union@tau={TAU:<18}{name:<12} (no captured plan)", flush=True)
            continue
        dirty, clean = _raha_pair(name)
        plan = _repairs_only(union_plans(verify_plan(dirty, base, tau=TAU),
                                         mock_plan(dirty)))
        cleaned, _ = apply_plan(dirty, plan)
        m = score(dirty, clean, cleaned)
        rows.append({**_row(name, m), "plan_source": src})
        print(f"  union@tau={TAU:<18}{name:<12} P={m['precision']:.3f} "
              f"cov={m['recall']:.3f} F1={m['f1']:.3f} dmg={m['damage']:.4f} "
              f"changed={m['_changed']} fixed={m['_fixed']}", flush=True)
    table["systems"]["verified_union_v6_tau0.5"] = {"per_dataset": rows}

    jelly = json.load(open(RESULTS / "jellyfish_raha.json"))
    rows = [{"dataset": name,
             **{k: jelly["per_dataset"][name][k]
                for k in ("f1", "precision", "recall", "damage")}}
            for name, _dom in RAHA]
    table["systems"]["jellyfish_ed_di"] = {
        "per_dataset": rows,
        "macro_f1": sum(r["f1"] for r in rows) / len(rows)}

    baran = json.load(open(RESULTS / "baran_raha.json"))
    by_ds: dict[str, list] = {}
    for r in baran["per_dataset"]:
        by_ds.setdefault(r["name"], []).append(r)
    rows = []
    for name, _dom in RAHA:
        rs = by_ds[name]
        rows.append({"dataset": name,
                     **{k: sum(r[k] for r in rs) / len(rs)
                        for k in ("f1", "precision", "recall", "damage")},
                     "n_seeds": len(rs)})
    table["systems"]["baran_oracle20"] = {
        "per_dataset": rows,
        "macro_f1": sum(r["f1"] for r in rows) / len(rows)}

    # acceptance check: grounded macro must match the money table's REAL-F1
    money = json.load(open(RESULTS / "money_table.json"))
    expect = next(r["real_f1"] for r in money if r["system"] == "grounded (ours)")
    got = table["systems"]["grounded"]["macro_f1"]
    ok = abs(got - expect) < 1e-9
    table["macro_check"] = {"grounded_macro_f1": got, "money_table_real_f1": expect,
                            "match": ok}
    print(f"\nmacro check: grounded {got:.6f} vs money-table {expect:.6f} "
          f"-> {'PASS' if ok else 'FAIL'}")

    out = RESULTS / "raha_per_dataset.json"
    json.dump(table, open(out, "w"), indent=1)
    print(f"written to {out}")
    print(latex(table))


LABELS = [("grounded", "Grounded (ours, deterministic)"),
          ("verified_union_v6_tau0.5", r"Verified union (v6, $\tau{=}0.5$)"),
          ("openrefine_fingerprint", "OpenRefine fingerprint"),
          ("openrefine_knn", "OpenRefine kNN"),
          ("jellyfish_ed_di", "Jellyfish-13B ED+DI"),
          ("baran_oracle20", r"Baran (oracle det.\ + 20 labels)")]


def latex(table: dict) -> str:
    """Booktabs rows: one block per system, one line per dataset."""
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"System & Dataset & Prec. & Rec. & F1 & Damage \\", r"\midrule"]
    for key, label in LABELS:
        sysrows = table["systems"][key]["per_dataset"]
        for i, r in enumerate(sysrows):
            head = label if i == 0 else ""
            if "missing" in r:
                L.append(f"{head} & {r['dataset'].replace('_', r'\_')} & "
                         r"\multicolumn{4}{c}{--- (no captured plan)} \\")
                continue
            L.append(f"{head} & {r['dataset'].replace('_', r'\_')} & "
                     f"{r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & "
                     f"{r['damage']:.3f} \\\\")
        macro = table["systems"][key].get("macro_f1")
        if macro is not None:
            L.append(f" & \\emph{{macro}} &  &  & \\emph{{{macro:.3f}}} &  \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
