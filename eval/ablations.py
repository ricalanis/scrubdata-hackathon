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


def main(seed: int = 7) -> None:
    print("\n=== Ablation suite (wide validation suite, single seed) — each removes ONE "
          "grounding component ===\n")
    print(f"{'variant':<28}{'NORTH*':>9}{'errtype':>9}{'domain':>8}{'damage':>9}{'abstain':>9}")
    print("-" * 72)
    rows = []
    for name, cfg in ABLATIONS:
        planner = (lambda df, c=cfg: mock_plan(df, ground_cfg=c))
        r = evaluate_suite(planner, seed=seed)
        rows.append((name, r))
        print(f"{name:<28}{r['north']:>9.3f}{r['et_macro']:>9.3f}{r['dom_macro']:>8.3f}"
              f"{r['damage']:>9.3f}{r['abstain']:>9.3f}")
    full = rows[0][1]
    print("\nDeltas vs full (what each component buys):")
    for name, r in rows[1:]:
        print(f"  {name:<28} ΔNORTH={r['north'] - full['north']:+.3f}  "
              f"Δdamage={r['damage'] - full['damage']:+.3f}  Δabstain={r['abstain'] - full['abstain']:+.3f}")
    print("\nGrounding lifts F1; abstain + ambiguity-check cut DAMAGE; case-match avoids "
          "convention damage. The combination is the contribution.")


if __name__ == "__main__":
    main()
