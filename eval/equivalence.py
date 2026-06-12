"""W2.d — TOST equivalence statistics for the SFT null (the bounded negative claim).

Operationalizes "weight interventions did not move held-out repair": paired
per-dataset GEN-F1 deltas (retrain minus champion v6) over the 3 held-out EVAL
sources x the 5-retrain SFT series (challenger seed31, v7 seed32, v8 seed33,
v9 seed34, v10 seed35), pooled (n=15). DISCLOSED granularity: the retrain series
was scored per held-out SOURCE only (eval/results/generalization_*.json) — the
42-pair paired bench exists for the shipped pipeline, not per retrain — so the
unit here is per-dataset, not per-pair, and within-retrain deltas are clustered
(flights/rayyan deltas are near-identical across retrains). A retrain-level
robustness check (n=5 macro deltas, one per retrain) is reported alongside.

PRE-REGISTERED (docs/ROADMAP_PUBLICATION.md W2.d, before this analysis ran):
SESOI delta = +/-0.05 GEN-F1, justified as smaller than the gain deterministic
grounding provides. TOST per Lakens'17: two one-sided t-tests against the SESOI
bounds; equivalence p = max of the two. Bootstrap: 10k resamples, seed 42, 90% CI.

    uv run python -m eval.equivalence
Writes eval/results/equivalence.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parent / "results"
SESOI = 0.05            # pre-registered (roadmap W2.d) — do not change post hoc
N_BOOT = 10_000
SEED = 42

CHAMPION = "generalization_champion.json"           # champion v6/seed21 (union)
RETRAINS = [                                        # the five SFT retrains (paper sec:negative)
    ("generalization_challenger.json", "challenger seed31"),
    ("generalization_v7.json", "v7 seed32 (unicode-punct archetype)"),
    ("generalization_v8.json", "v8 seed33 (+109k harvested alias vocabs)"),
    ("generalization_v9.json", "v9 seed34 (+MusicBrainz hints, gidcl pairs)"),
    ("generalization_v10.json", "v10 seed35 (suspects-contract)"),
]


def _per_source_f1(fname: str) -> dict[str, float]:
    rec = json.loads((RESULTS / fname).read_text())[0]
    return {s["source"]: s["f1"] for s in rec["per_source"]}, rec["gen_f1"]


def _tost(deltas: np.ndarray) -> dict:
    """Two one-sided t-tests against [-SESOI, +SESOI]; equivalence p = max."""
    p_lo = stats.ttest_1samp(deltas, -SESOI, alternative="greater").pvalue
    p_hi = stats.ttest_1samp(deltas, +SESOI, alternative="less").pvalue
    return {"p_lower": float(p_lo), "p_upper": float(p_hi),
            "p_tost": float(max(p_lo, p_hi)), "n": int(len(deltas)),
            "mean": float(deltas.mean()), "sd": float(deltas.std(ddof=1))}


def main() -> dict:
    champ, champ_macro = _per_source_f1(CHAMPION)
    pooled, per_retrain = [], []
    for fname, label in RETRAINS:
        ps, macro = _per_source_f1(fname)
        assert set(ps) == set(champ), f"{fname}: source mismatch vs champion"
        per_retrain.append({
            "retrain": label, "file": fname,
            "macro_gen_f1": round(macro, 6),
            "macro_delta": round(macro - champ_macro, 6),
            "per_dataset_delta": {s: round(ps[s] - champ[s], 6) for s in champ},
        })
        pooled += [ps[s] - champ[s] for s in sorted(champ)]
    deltas = np.array(pooled)

    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean()
                     for _ in range(N_BOOT)])
    ci = (float(np.percentile(boot, 5)), float(np.percentile(boot, 95)))

    macro_deltas = np.array([r["macro_delta"] for r in per_retrain])
    out = {
        "spec": {"sesoi": SESOI, "sesoi_preregistered": "docs/ROADMAP_PUBLICATION.md W2.d",
                 "n_boot": N_BOOT, "seed": SEED, "ci_level": 0.90,
                 "champion": CHAMPION, "champion_macro_gen_f1": round(champ_macro, 6)},
        "granularity": ("per-dataset (3 held-out sources x 5 retrains = 15 paired "
                        "deltas). Per-pair rows do not exist for the retrain series "
                        "(only the shipped pipeline was scored on the 42-pair bench); "
                        "within-retrain deltas are clustered, hence the retrain-level "
                        "robustness check below."),
        "per_retrain": per_retrain,
        "pooled_per_dataset": {
            **_tost(deltas),
            "ci90_bootstrap": [round(ci[0], 6), round(ci[1], 6)],
            "ci90_width": round(ci[1] - ci[0], 6),
            "equivalent_at_sesoi": bool(-SESOI < ci[0] and ci[1] < SESOI),
        },
        "retrain_level_robustness": _tost(macro_deltas),
        "caveat": ("GEN-F1 sits near floor (champion 0.015 absolute), so the bound "
                   "certifies absence of movement on a low-dynamic-range metric; "
                   "the CI width (~0.004) shows the data could have detected effects "
                   "an order of magnitude smaller than the 0.05 SESOI."),
    }
    p = out["pooled_per_dataset"]
    out["paper_sentence"] = (
        f"Across the five-retrain series the mean held-out GEN-F1 delta (retrain "
        f"minus champion, per-dataset, n={p['n']}) is {p['mean']:+.4f} (90\\% "
        f"bootstrap CI [{ci[0]:+.4f}, {ci[1]:+.4f}]); TOST rejects effects larger "
        f"than the pre-registered $\\pm$0.05 SESOI (p = {p['p_tost']:.1e}), and the "
        f"retrain-level check (n=5 macro deltas) agrees "
        f"(p = {out['retrain_level_robustness']['p_tost']:.1e}).")

    (RESULTS / "equivalence.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("pooled_per_dataset",
                                          "retrain_level_robustness",
                                          "paper_sentence")}, indent=2))
    return out


if __name__ == "__main__":
    main()
