"""D1 — the GENERALIZATION metric: held-out-source real-error evaluation.

The wide-suite REAL slice mixes sources whose pairs are IN the champion's training mix
(hospital/beers/movies_1 -> mixA), so it part-measures memorization. This metric fixes
that and one more honesty problem:

  * HELD-OUT SOURCES ONLY: a model is scored only on real-error benchmarks whose pairs
    were never used to train it. The split is explicit and committed (TRAIN_SOURCES);
    new harvested sources must be assigned to exactly one side.
  * ERROR-CLASS BREAKDOWN: benchmark errors split by the SAME variant gate the training
    derivation uses (training.real_data._is_variant — one source of truth). A
    canonicalization system claims competence on the VARIANT class (typos / casing /
    aliases); imputation-class errors (missing or non-variant rewrites) are reported,
    never hidden, but a system that abstains on them is behaving correctly.

Headline numbers per system:
    GEN-F1          churn-neutral F1 over ALL errors, macro over held-out sources
    VARIANT-RECALL  share of variant-class errors repaired (claimed competence)
    VARIANT-PREC    of committed changes on variant cells, share correct
    damage          clean cells corrupted (churn-neutral)

    uv run python -m eval.generalization                 # grounded heuristic baseline
"""

from __future__ import annotations

import argparse
import json

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan
from training.real_data import _is_variant

from .metrics import _cell_equal
from .run_real_multi import _cell_only, _fetch, _sem_equal, score

# pairs used to train the current champion (v6 = mixA) — anything here is OFF-LIMITS
# for generalization scoring of that model. Update per training run.
TRAIN_SOURCES = {"v6": {"hospital", "beers", "movies_1"}}

# held-out real-error sources. Harvested D1 sources get appended here OR to the
# training side — never both. ed2_restaurants (stage-2 harvest): real NYC-restaurant
# typos, in-regime, EVAL-ONLY — its sibling domain source (fodors_zagats) trains, so
# this measures cross-source same-domain transfer. dblp_scholar was REJECTED as an
# eval source: its gold systematically prefers the opposite case convention from the
# dirty side (Scholar lowercase vs DBLP Title Case), which measures convention
# preference, not cleaning — the artifact this metric is designed against.
EVAL_SOURCES = ["flights", "rayyan", "ed2_restaurants"]


def variant_breakdown(dirty, clean, out) -> dict:
    """Split benchmark errors by class and count repairs per class (churn-neutral)."""
    n = min(len(dirty), len(out), len(clean))
    c = {"variant_errors": 0, "variant_fixed": 0, "variant_changed": 0,
         "variant_good": 0, "other_errors": 0, "other_fixed": 0}
    for j, col in enumerate(dirty.columns):
        present = col in out.columns
        for i in range(n):
            dv, cv = dirty.iat[i, j], clean.iat[i, j]
            if _cell_equal(dv, cv):
                continue                                   # not a benchmark error
            ov = out.iloc[i][col] if present else dv
            chg = present and not _cell_equal(ov, dv)
            if chg and _sem_equal(ov, dv) and not _cell_equal(ov, cv):
                chg = False                                # churn: ignore
            fixed = _cell_equal(ov, cv) or (_sem_equal(ov, cv) and chg)
            is_variant = (str(dv).strip() and str(cv).strip()
                          and _is_variant(str(dv), str(cv)))
            if is_variant:
                c["variant_errors"] += 1
                c["variant_fixed"] += int(fixed)
                if chg:
                    c["variant_changed"] += 1
                    c["variant_good"] += int(_sem_equal(ov, cv))
            else:
                c["other_errors"] += 1
                c["other_fixed"] += int(fixed)
    return c


def evaluate_generalization(planner, sources=None, label: str = "system") -> dict:
    sources = sources or EVAL_SOURCES
    rows = []
    for name in sources:
        # FULL tables, no truncation — ed2_restaurants' real errors are concentrated
        # outside the first 2k rows (_raha_pair's head(2000) hid 473 of 477).
        dirty, clean = _fetch(name)
        cleaned, _ = apply_plan(dirty, _cell_only(planner(dirty)))
        m = score(dirty, clean, cleaned)
        b = variant_breakdown(dirty, clean, cleaned)
        rows.append({"source": name, **{k: m[k] for k in
                                        ("f1", "precision", "recall", "damage")}, **b})
        print(f"  {name:<10} F1={m['f1']:.3f} dmg={m['damage']:.3f} | variant: "
              f"{b['variant_fixed']}/{b['variant_errors']} fixed, "
              f"{b['variant_good']}/{b['variant_changed']} changes good | "
              f"other: {b['other_fixed']}/{b['other_errors']}", flush=True)

    return _aggregate(rows, sources, label)


def evaluate_captured_union(plans: dict, sources, label: str, tau: float = 0.5) -> dict:
    """Score the SHIPPED pipeline from captured raw model plans (Modal --capture):
    per source, verify(tau) the captured plan, union with the grounded heuristic —
    byte-identical composition to scrubdata/active.py."""
    from scrubdata.verifier import union_plans, verify_plan

    def planner_for(name):
        def planner(df, *_):
            return union_plans(verify_plan(df, plans[name], tau=tau), mock_plan(df))
        return planner

    rows = []
    for name in sources:
        dirty, clean = _fetch(name)
        cleaned, _ = apply_plan(dirty, _cell_only(planner_for(name)(dirty)))
        m = score(dirty, clean, cleaned)
        b = variant_breakdown(dirty, clean, cleaned)
        rows.append({"source": name, **{k: m[k] for k in
                                        ("f1", "precision", "recall", "damage")}, **b})
        print(f"  {name:<16} F1={m['f1']:.3f} dmg={m['damage']:.3f} | variant: "
              f"{b['variant_fixed']}/{b['variant_errors']} fixed", flush=True)
    return _aggregate(rows, sources, label)


def _aggregate(rows, sources, label) -> dict:
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    def rate(num, den):
        return num / den if den else 0.0

    out = {
        "system": label, "sources": list(sources),
        "gen_f1": mean(r["f1"] for r in rows),
        "variant_recall": mean(rate(r["variant_fixed"], r["variant_errors"]) for r in rows),
        "variant_precision": mean(rate(r["variant_good"], r["variant_changed"])
                                  if r["variant_changed"] else 1.0 for r in rows),
        "other_recall": mean(rate(r["other_fixed"], r["other_errors"]) for r in rows),
        "damage": mean(r["damage"] for r in rows),
        "per_source": rows,
    }
    print(f"{label}: GEN-F1={out['gen_f1']:.3f} VARIANT-RECALL={out['variant_recall']:.3f} "
          f"VARIANT-PREC={out['variant_precision']:.3f} dmg={out['damage']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(EVAL_SOURCES))
    ap.add_argument("--plans", default=None,
                    help="JSON file {source: captured raw model plan} -> score the "
                         "shipped union pipeline instead of the local baselines")
    ap.add_argument("--label", default="captured union")
    ap.add_argument("--out", default="eval/results/generalization_baseline.json")
    args = ap.parse_args()
    sources = args.sources.split(",")
    if args.plans:
        plans = json.load(open(args.plans))
        results = [evaluate_captured_union(plans, sources, args.label)]
    else:
        results = [
            evaluate_generalization(mock_plan, sources, "grounded heuristic"),
            evaluate_generalization(
                lambda df: {"table_operations": [], "columns": [], "flags": []},
                sources, "no-op"),
        ]
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
