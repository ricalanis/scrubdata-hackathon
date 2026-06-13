# Degenerate baselines + cost-weighted damage (W4.3 + W4.4)

Same 42 dirty/clean pairs as `eval/paired_bench.py`, scored with `run_real_multi.score()` (churn-neutral F1 + damage). The degenerate policies pin
the metric: no-op = floor (F1 0, damage 0), oracle = ceiling (F1 1, damage 0),
random-edit (seeded, 5% of cells) = vandalism the metric must punish. Abstain-all
is score-identical to no-op — the repair metric is flag-blind by design.

| policy | macro F1 | macro P | macro R | macro damage | fixed | damage cells |
|---|---|---|---|---|---|---|
| no-op | 0.000 | 1.000 | 0.000 | 0.0000 | 0 | 0 |
| abstain-all | 0.000 | 1.000 | 0.000 | 0.0000 | 0 | 0 |
| random-edit | 0.000 | 0.001 | 0.001 | 0.0485 | 39 | 80042 |
| oracle | 1.000 | 1.000 | 1.000 | 0.0000 | 163607 | 0 |
| shipped | 0.343 | 0.576 | 0.308 | 0.0229 | 83543 | 61679 |

## Cost-weighted scores (Effective-Reliability style, W4.4)

score_c = fixes − c·damage_cells, micro-summed over all pairs; per-error =
score_c / 163607 total benchmark errors.

| policy | c=1 (per-error) | c=5 (per-error) | c=10 (per-error) |
|---|---|---|---|
| no-op | 0 (+0.000) | 0 (+0.000) | 0 (+0.000) |
| abstain-all | 0 (+0.000) | 0 (+0.000) | 0 (+0.000) |
| random-edit | -80003 (-0.489) | -400171 (-2.446) | -800381 (-4.892) |
| oracle | 163607 (+1.000) | 163607 (+1.000) | 163607 (+1.000) |
| shipped | 21864 (+0.134) | -224852 (-1.374) | -533247 (-3.259) |

Acceptance: oracle F1 = 1.0 on all pairs: **True** · no-op damage = 0.0 on all pairs: **True**
Repro: `uv run python -m eval.degenerate` (seed 7, edit fraction 0.05).
