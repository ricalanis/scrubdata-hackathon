"""W4.3 + W4.4 — degenerate baselines + cost-weighted damage over the paired sets.

Four scorer-pinning policies over the same dirty/clean pairs eval/paired_bench.py
walks: no-op (output = dirty), abstain-all (no-op + flags; score-identical at the
cell level — the repair metric is flag-blind by design, flags surface in audit
metrics), random-edit (seeded vandalism: 5% of cells replaced with another value
from the same column) and oracle (output = clean, headers realigned to dirty's —
23/42 pairs differ in header naming only; cell alignment is positional). They pin
the metric's floor (no-op F1 = 0, damage = 0), ceiling (oracle F1 = 1, damage = 0)
and show it punishes vandalism. Also reruns the SHIPPED pipeline (mock_plan) to
capture raw fix/damage cell counts and reports Effective-Reliability-style
cost-weighted scores score_c = fixes - c*damage_cells for c in {1, 5, 10}.

    uv run python -m eval.degenerate
Writes eval/results/degenerate.json + docs/DEGENERATE_BASELINES.md. Per-pair rows
are cached incrementally (eval/results/degenerate_pairs.json) so a killed run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .paired_bench import _load, pairs
from .run_real_multi import _cell_only, score

ROOT = Path(__file__).resolve().parent.parent
EDIT_FRAC = 0.05
SEED = 7
COSTS = (1, 5, 10)


def _noop(dirty, clean):
    return dirty


def _abstain_all(dirty, clean):
    return dirty.copy()          # + flags conceptually; the cell metric is flag-blind


def _random_edit(dirty, clean, seed=SEED):
    rng = random.Random(seed)
    out = dirty.copy()
    n, m = out.shape
    uniq = [list(dict.fromkeys(out.iloc[:, j])) for j in range(m)]
    for idx in rng.sample(range(n * m), max(1, int(n * m * EDIT_FRAC))):
        i, j = divmod(idx, m)
        alts = [v for v in uniq[j] if v != out.iat[i, j]]
        if alts:
            out.iat[i, j] = rng.choice(alts)
    return out


def _oracle(dirty, clean):
    out = clean.copy()
    out.columns = dirty.columns  # header-naming variants only; alignment is positional
    return out


def _shipped(dirty, clean):
    return apply_plan(dirty, _cell_only(mock_plan(dirty)))[0]


POLICIES = [("no-op", _noop), ("abstain-all", _abstain_all),
            ("random-edit", _random_edit), ("oracle", _oracle),
            ("shipped", _shipped)]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="eval/results/degenerate.json")
    ap.add_argument("--cache", default="eval/results/degenerate_pairs.json")
    args = ap.parse_args()
    cache = json.load(open(args.cache)) if Path(args.cache).exists() else {}
    for p in pairs():
        if args.only and p.name != args.only:
            continue
        if p.name in cache:
            continue
        try:
            dirty, clean = _load(p)
        except Exception as e:  # noqa: BLE001
            print(f"  {p.name}: LOAD FAILED {type(e).__name__}")
            continue
        entry = {}
        for name, policy in POLICIES:
            t0 = time.perf_counter()
            m = score(dirty, clean, policy(dirty, clean))
            n = min(len(dirty), len(clean))
            clean_cells = n * dirty.shape[1] - m["_errors"]
            entry[name] = {
                "name": p.name, "errors": m["_errors"],
                "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "damage": m["damage"], "fixed": m["_fixed"], "changed": m["_changed"],
                "damage_cells": round(m["damage"] * clean_cells),
                "sec": round(time.perf_counter() - t0, 1)}
        cache[p.name] = entry
        json.dump(cache, open(args.cache, "w"), indent=1)
        print(f"  {p.name:<46} " + " ".join(
            f"{name}={entry[name]['f1']:.3f}" for name, _ in POLICIES), flush=True)
    res = {name: [cache[k][name] for k in sorted(cache)] for name, _ in POLICIES}

    out = {"n_pairs": len(res["no-op"]), "edit_frac": EDIT_FRAC, "seed": SEED,
           "policies": {}, "acceptance": {}}
    for name, _ in POLICIES:
        rows = res[name]
        E, F, D = (sum(r[k] for r in rows) for k in ("errors", "fixed", "damage_cells"))
        out["policies"][name] = {
            "macro": {k: round(_mean(r[k] for r in rows), 4)
                      for k in ("f1", "precision", "recall", "damage")},
            "micro": {"errors": E, "fixed": F, "changed": sum(r["changed"] for r in rows),
                      "damage_cells": D},
            "score_c": {f"c={c}": {"raw": F - c * D,
                                   "per_error": round((F - c * D) / E, 4)}
                        for c in COSTS},
            "sec": round(sum(r["sec"] for r in rows), 1),
            "per_pair": rows}
    bad_oracle = [r["name"] for r in res["oracle"] if r["f1"] != 1.0]
    bad_noop = [r["name"] for r in res["no-op"] if r["damage"] != 0.0]
    out["acceptance"] = {"oracle_f1_all_exactly_1": not bad_oracle,
                         "noop_damage_all_exactly_0": not bad_noop,
                         "violations": {"oracle": bad_oracle, "no-op": bad_noop}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    P = out["policies"]
    L = ["# Degenerate baselines + cost-weighted damage (W4.3 + W4.4)", "",
         f"Same {out['n_pairs']} dirty/clean pairs as `eval/paired_bench.py`, scored with "
         "`run_real_multi.score()` (churn-neutral F1 + damage). The degenerate policies pin",
         "the metric: no-op = floor (F1 0, damage 0), oracle = ceiling (F1 1, damage 0),",
         "random-edit (seeded, 5% of cells) = vandalism the metric must punish. Abstain-all",
         "is score-identical to no-op — the repair metric is flag-blind by design.", "",
         "| policy | macro F1 | macro P | macro R | macro damage | fixed | damage cells |",
         "|---|---|---|---|---|---|---|"]
    for name, _ in POLICIES:
        ma, mi = P[name]["macro"], P[name]["micro"]
        L.append(f"| {name} | {ma['f1']:.3f} | {ma['precision']:.3f} | {ma['recall']:.3f} "
                 f"| {ma['damage']:.4f} | {mi['fixed']} | {mi['damage_cells']} |")
    L += ["", "## Cost-weighted scores (Effective-Reliability style, W4.4)", "",
          "score_c = fixes − c·damage_cells, micro-summed over all pairs; per-error =",
          f"score_c / {P['shipped']['micro']['errors']} total benchmark errors.", "",
          "| policy | " + " | ".join(f"c={c} (per-error)" for c in COSTS) + " |",
          "|---|" + "---|" * len(COSTS)]
    for name, _ in POLICIES:
        sc = P[name]["score_c"]
        L.append(f"| {name} | " + " | ".join(
            f"{sc[f'c={c}']['raw']} ({sc[f'c={c}']['per_error']:+.3f})" for c in COSTS) + " |")
    a = out["acceptance"]
    L += ["", f"Acceptance: oracle F1 = 1.0 on all pairs: **{a['oracle_f1_all_exactly_1']}** · "
          f"no-op damage = 0.0 on all pairs: **{a['noop_damage_all_exactly_0']}**",
          f"Repro: `uv run python -m eval.degenerate` (seed {SEED}, edit fraction {EDIT_FRAC})."]
    (ROOT / "docs" / "DEGENERATE_BASELINES.md").write_text("\n".join(L) + "\n")
    print(f"{out['n_pairs']} pairs x {len(POLICIES)} policies -> {args.out} "
          "+ docs/DEGENERATE_BASELINES.md")
    print("acceptance:", out["acceptance"])


if __name__ == "__main__":
    main()
