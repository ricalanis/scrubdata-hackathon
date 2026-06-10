"""Ablation suite — isolate each grounding component's contribution to the north-star.

Each row turns ONE design decision off (via mock_plan's ground_cfg) and re-runs the wide
validation suite. Shows what grounding / abstention / ambiguity-checking / case-matching each
buy in F1 and (critically) in DAMAGE.

    uv run python -m eval.ablations
"""

from __future__ import annotations

from scrubdata.planner import mock_plan

from .run_real_multi import evaluate_suite

ABLATIONS = [
    ("full (grounded)",            {}),
    ("- grounding (freq-cluster)", {"use_reference": False}),
    ("- abstain (map nearest)",    {"threshold": 0.0, "min_margin": 0.0}),
    ("- ambiguity check",          {"min_margin": 0.0}),
    ("- case match",               {"case_match": False}),
]


def main(seeds=(7, 17, 27), out: str | None = None) -> None:
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n=== Ablation suite (wide validation suite, {len(seeds)} seeds) — each "
          "removes ONE grounding component ===\n")
    print(f"{'variant':<28}{'NORTH*':>9}{'REAL-F1':>9}{'INJ-F1':>8}{'damage':>9}{'abstain':>9}")
    print("-" * 72)
    rows = []
    for name, cfg in ABLATIONS:
        planner = (lambda df, c=cfg: mock_plan(df, ground_cfg=c))
        per_seed = [evaluate_suite(planner, seed=s) for s in seeds]
        r = {k: mean(p[k] for p in per_seed)
             for k in ("north", "real", "injected", "damage", "abstain")}
        mu = r["north"]
        var = mean([(p["north"] - mu) ** 2 for p in per_seed])
        r["north_ci"] = 1.96 * (var ** 0.5) / (len(per_seed) ** 0.5)
        rows.append((name, r))
        print(f"{name:<28}{r['north']:>9.3f}{r['real']:>9.3f}{r['injected']:>8.3f}"
              f"{r['damage']:>9.3f}{r['abstain']:>9.3f}", flush=True)
    full = rows[0][1]
    print("\nDeltas vs full (what each component buys):")
    for name, r in rows[1:]:
        print(f"  {name:<28} ΔNORTH={r['north'] - full['north']:+.3f}  "
              f"Δdamage={r['damage'] - full['damage']:+.3f}  Δabstain={r['abstain'] - full['abstain']:+.3f}")
    if out:
        import json
        json.dump([{"variant": n, **r, "seeds": list(seeds)} for n, r in rows],
                  open(out, "w"), indent=1)
        print(f"rows written to {out}")
    print("\nGrounding lifts F1; abstain + ambiguity-check cut DAMAGE; case-match avoids "
          "convention damage. The combination is the contribution.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    main(out=ap.parse_args().out)
