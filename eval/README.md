# Eval harness + goalpost

Measures any planner against a **held-out** synthetic gold set (seed differs from
training, and gold is filtered to oracle-solvable so the ceiling is a clean 1.0).

```bash
uv run eval/run_eval.py --n 300 --seed 4242
```

Adopts the researched tooling: `jsonschema` for plan validity; set-based micro-F1 for
operations and canonicalization mappings; the **executor itself** for end-to-end
cell-recovery (the Raha-style dirty→clean comparison). promptfoo + `llm-rubric` will
wrap the report-quality layer once a model exists.

## Metrics
- **json_valid** — plan conforms to the schema (`eval/metrics.py:PLAN_SCHEMA`).
- **op_f1 / op_r** — micro-F1 / recall over `(column, operation)` pairs vs gold.
- **canon_f1 / canon_r** — micro-F1 / recall over `(column, raw→canonical)` mapping
  pairs. *This is the fuzzy skill rules can't do — the whole reason for the model.*
- **recovery** — fraction of clean-reference cells recovered by executing the plan.

## Baseline (measured) and the goalpost

Two reference systems frame every run:
- **ORACLE** = the gold plan → the ceiling.
- **HEURISTIC** (`scrubdata.mock_plan`) = the rule-based baseline the model must beat.

Measured on the frozen 300-example gold set (`eval/gold.jsonl`, **value_counts/aggregation
format**):

| system | json_valid | op_f1 | canon_f1 | canon_r | recovery |
|---|---|---|---|---|---|
| ORACLE (gold) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| HEURISTIC (baseline) | 1.000 | 0.932 | **0.189** | 0.129 | **0.637** |

**Reading:** with case-folding + typo-clustering the heuristic does the *easy*
canonicalization (collapse to most-frequent surface), but it's still ~blind to
**alias/semantic** canonicalization (`USA`→`United States`, `NYC`→`New York`) — canon_f1
0.19 vs the oracle's 1.0. That gap is the fine-tuned model's job. (Earlier, on the old
sample-rows format, a fine-tune reached canon_f1 0.86 vs a big vanilla model's 0.45 —
proving small-aligned > big-generic; the v4 retrain re-establishes this on the new format.)

### 🎯 Goalpost for the fine-tuned Qwen3-4B
| metric | baseline | **target** | ceiling |
|---|---|---|---|
| json_valid | 1.000 | **≥ 0.99** | 1.000 |
| op_f1 | 0.932 | **≥ 0.98** | 1.000 |
| canon_f1 | 0.189 | **≥ 0.85** | 1.000 |
| recovery | 0.637 | **≥ 0.95** | 1.000 |

A fine-tune that hits these clearly beats the (now stronger) heuristic and approaches the
oracle — the headline being **canon_f1 0.133 → ≥0.85** (alias-level canonicalization) and
**recovery 0.627 → ≥0.95**.

## Plugging in the model
`evaluate(planner, gold)` takes any `planner(dirty_df, gold_plan) -> plan dict`. For
the model, wrap inference (build prompt via `scrubdata.prompt`, parse JSON) and pass it
in alongside the two reference systems. Track the table every fine-tune iteration; the
per-metric delta vs baseline is the cheap regression signal.

## Layer 2 — real out-of-distribution data (`uv run eval/run_real.py`)

Raha `hospital` (1000×20, row-aligned dirty/clean). Errors are char-substitution typos
(`birminghxm`→`birmingham`) — only ~2.5% of cells. Scored with the Raha **repair**
protocol (the right metric when data is already mostly correct):

| system | recovery | repair_recall | repair_prec | broken |
|---|---|---|---|---|
| NO-OP (dirty as-is) | 0.975 | 0.000 | 0.000 | 0 |
| HEURISTIC (baseline) | 0.880 | **0.293** | 0.065 | 2041 |

(Typo-clustering now fixes ~29% of the real char-substitution errors — up from 0. The
model should push repair_recall higher and improve repair_prec.)

**Reading (honest + important):** the rule heuristic fixes **0** typos. Its 2021 changed
cells are **convention divergence, not errors** — our tool parses `100%`→`1.0` and
reformats phones, which this benchmark stores as raw text. That's product value, so raw
`recovery`/`broken` *understates* a standardizing tool on a foreign benchmark. The honest
metric here is **`repair_recall`** — did we fix the actual char-substitution typos
(`birminghxm`→`birmingham`)? The heuristic can't (scores 0); cluster-canonicalization is
the model's job. Two takeaways:
1. **The headline real-data metric is `repair_recall`** (error-fixing), not recovery.
2. **Product feature surfaced:** offer a "preserve original formats" toggle — some users
   want raw representation kept; standardizing is the default but should be reversible
   (matches PRODUCT.md's trust contract).

### 🎯 Real-data goalpost (fine-tuned model)
| metric | NO-OP | HEURISTIC | **target** | note |
|---|---|---|---|---|
| **repair_recall** | 0.000 | 0.000 | **≥ 0.30** | the real test — fix typos via clustering |
| repair_prec | 0.000 | 0.000 | **≥ 0.70** | of cells changed, fraction that fixed an error |
| recovery | 0.975 | 0.874 | report-only | convention-sensitive; not a pass/fail gate |

The model plugs into `_score(dirty, clean, model_output)` exactly like the heuristic.

> Data auto-fetched to `data/real/hospital/` (gitignored). Add Flights/Beers/CleanML the
> same way for breadth.

## Scale: aggregation + agentic batching (validated)

Cleaning *large* tables doesn't mean bigger prompts — it means reasoning over **patterns**:
- **Aggregation** — the profiler sends per-column `value_counts` (`[value, frequency]`), so
  the prompt size depends on DISTINCT values, not rows. Rare typos sit at the tail next to
  their dominant canonical (`birminghxm`:1 vs `birmingham`:312) — visible at any scale.
- **Column batching** — `scrubdata.model_planner.make_batched_planner` plans a wide table
  in small column-batches, so a 20-column table never blows one prompt.

**Validated** on the real Raha hospital table (1000×20) with a *vanilla* model (no retrain):
**repair_recall 0.509** (fixed 259/509 typos), vs **0.000** for the old one-shot+sample-rows
approach. The v4 fine-tune trains on this `value_counts` format.

---

## The wide suite (current north-star)

The single-dataset hospital metric was retired as north-star (biased: one table,
recall-only, convention-sensitive, abstain-blind). The current harness:

- **`run_real_multi.py`** — 65-dataset suite (5 Raha real-error benchmarks + seeded
  error injection over 15 harvested open-data domains), scored with a **churn-neutral**
  metric (pure case/whitespace rewrites that don't restore gold count as nothing) and
  aggregated as a **double macro** (error-type × domain, harmonic mean) so no single
  table or error type dominates. Reports REAL vs INJECTED slices separately — injected
  typos are in-distribution for frequency clustering by construction.
- **`ablations.py`** — removes one grounding component at a time (reference, abstain,
  ambiguity margin, case-match). Caught two metric artifacts (churn inflation,
  reference-unsafe traps) now fixed and documented in the paper.
- **`calibration.py`** — risk–coverage + ECE for the abstention confidence
  (AURC 0.120; 90% precision at the default threshold, ≥95% at 0.91).
- **`pii_leak.py`** — masking leak test: 0/360 residual detectable PII.
- **`pii_slice.py`** — OOD PII column typing on Gretel test: 5/5 types, 0/7 FP.
- **`inject.py`** — seeded, self-verifying error injectors (typo/OCR/case/whitespace)
  that turn any clean table into validation data.

Baselines include OpenRefine fingerprint + kNN clustering (`scrubdata/baselines.py`,
with blocking, as the real tool uses). Full results & discussion: `docs/paper/`.
