"""Paper figure: plan-level precision-coverage on hospital (Figure fig:pc).

Reads the curve artifacts produced by eval/precision_curve.py and renders
docs/paper/fig_precision_coverage.{pdf,png}.

    uv run --with matplotlib python -m eval.plot_precision_curve
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "docs" / "paper"
RAW_MODEL = (0.4754, 0.1849)            # unverified v6 plan: coverage, precision
SHIP_TAU = 0.5


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = json.load(open(RESULTS / "v6_hospital_precision_curve.json"))
    union = json.load(open(RESULTS / "v6_hospital_union_curve.json"))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for fname in ("v6s25_hospital_union_curve.json", "v6s26_hospital_union_curve.json"):
        p = RESULTS / fname
        if p.exists():                       # sibling training seeds, faint
            rows = json.load(open(p))
            pts = sorted(((r["coverage"], r["precision"]) for r in rows if r["tau"] > 0))
            ax.plot([x for x, _ in pts], [y for _, y in pts], lw=0.9, color="#2f6f5e",
                    alpha=0.25, zorder=1,
                    label="union, sibling seeds" if "s25" in fname else None)
    for rows, label, color, marker in (
            (model, "verifier-gated model plan", "#888888", "s"),
            (union, "verified union (shipped)", "#2f6f5e", "o")):
        pts = sorted(((r["coverage"], r["precision"], r["tau"]) for r in rows
                      if r["tau"] > 0), key=lambda p: p[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, ms=4,
                lw=1.6, color=color, label=label)
    ship = next(r for r in union if r["tau"] == SHIP_TAU)
    ax.scatter([ship["coverage"]], [ship["precision"]], s=90, zorder=5,
               facecolor="none", edgecolor="#2f6f5e", lw=2)
    ax.annotate(f"shipped ($\\tau$=0.5)\n{ship['precision']:.3f} @ {ship['coverage']:.3f}",
                xy=(ship["coverage"], ship["precision"]),
                xytext=(ship["coverage"] - 0.155, ship["precision"] - 0.18),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.scatter([RAW_MODEL[0]], [RAW_MODEL[1]], s=50, color="#b3433b", marker="x", zorder=5)
    ax.annotate(f"raw model plan\n{RAW_MODEL[1]:.3f} @ {RAW_MODEL[0]:.3f}",
                xy=RAW_MODEL, xytext=(RAW_MODEL[0] - 0.13, RAW_MODEL[1] + 0.12),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.axhline(0.90, ls=":", lw=0.8, color="#999999")
    ax.text(0.012, 0.905, "0.90 precision", fontsize=7, color="#777777")
    ax.set_xlabel("coverage (share of 509 real errors repaired)")
    ax.set_ylabel("precision (committed changes correct)")
    ax.set_xlim(0, 0.62)
    ax.set_ylim(0, 1.04)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_precision_coverage.{ext}", dpi=200)
    print(f"wrote {OUT}/fig_precision_coverage.(pdf|png)")


if __name__ == "__main__":
    main()
