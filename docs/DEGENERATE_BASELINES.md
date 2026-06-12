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
score_c / 163610 total benchmark errors.

| policy | c=1 (per-error) | c=5 (per-error) | c=10 (per-error) |
|---|---|---|---|
| no-op | 0 (+0.000) | 0 (+0.000) | 0 (+0.000) |
| abstain-all | 0 (+0.000) | 0 (+0.000) | 0 (+0.000) |
| random-edit | -80003 (-0.489) | -400171 (-2.446) | -800381 (-4.892) |
| oracle | 163607 (+1.000) | 163607 (+1.000) | 163607 (+1.000) |
| shipped | 21864 (+0.134) | -224852 (-1.374) | -533247 (-3.259) |

Acceptance: oracle F1 = 1.0 on all pairs: **False** · no-op damage = 0.0 on all pairs: **True**
Repro: `uv run python -m eval.degenerate` (seed 7, edit fraction 0.05).

## Acceptance investigation: the one oracle violation is a metric bug (found by this check)

Oracle F1 = 1.0 exactly on 41/42 pairs; on `zeroed_tax100k` it is 0.9984 (949/952).
Diagnosis: the 3 missed cells hold the literal string `Nan` (a first name, column
`f_name`) in BOTH dirty and clean. In `eval/metrics.py::_cell_equal`, `float("Nan")`
parses to NaN and `math.isclose(nan, nan)` is False, so the string-equality fallback
is never reached — any value parsing to float NaN is unequal to ITSELF. Effects:
(a) 3 phantom "errors" where dirty == clean; (b) the oracle's restoration of them is
churn-zeroed (sem-equal to input, raw-unequal), so no policy can ever "fix" them.
Blast radius (scanned all 42 pairs, 1,786,510 cells): exactly those 3 cells.
Verified fix (runtime-patched, not yet applied to `eval/metrics.py` — it is shared by
every published number): inside the `try`, return `str(a) == str(b)` when either
side parses to NaN. With it, `zeroed_tax100k` has 949 true errors, oracle F1 = 1.0
exactly, no-op unchanged. Shipped numbers shift by < 1e-4 (3 cells / 1.79M).

Determinism cross-check: the shipped rerun here reproduces
`eval/results/paired_bench.json` per-pair F1 on all 42 pairs (macro 0.3428 both).
