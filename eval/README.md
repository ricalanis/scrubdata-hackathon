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

Measured on 300 held-out examples (seed 4242):

| system | json_valid | op_f1 | canon_f1 | canon_r | recovery |
|---|---|---|---|---|---|
| ORACLE (gold) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| HEURISTIC (baseline) | 1.000 | 0.845 | **0.002** | 0.001 | **0.665** |

**Reading:** the heuristic leaves ~1/3 of cells wrong and is ~blind to
canonicalization. That 0.665 → 1.000 recovery gap is exactly the value a fine-tuned
model must deliver.

### 🎯 Goalpost for the fine-tuned Qwen3-4B
| metric | baseline | **target** | ceiling |
|---|---|---|---|
| json_valid | 1.000 | **≥ 0.99** | 1.000 |
| op_f1 | 0.845 | **≥ 0.95** | 1.000 |
| canon_f1 | 0.002 | **≥ 0.85** | 1.000 |
| recovery | 0.665 | **≥ 0.95** | 1.000 |

A fine-tune that hits these clearly beats the free heuristic and approaches the oracle
— the headline being **canon_f1 0.002 → ≥0.85** and **recovery 0.665 → ≥0.95**.

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
| HEURISTIC (baseline) | 0.874 | 0.000 | 0.000 | **2021** |

**Reading (honest + important):** the rule heuristic fixes **0** typos and **breaks 2021
good cells** by over-standardizing — ~1000 reformatting phones (the dataset's clean
convention keeps raw digits) and ~1000 collapsing distinct-but-valid categories. Two
takeaways:
1. **Model headroom on OOD is large** — it should *exceed* NO-OP by clustering typos
   (`birminghxm`→`birmingham`) while NOT reformatting/over-collapsing.
2. **Product requirement surfaced:** be conservative — only canonicalize genuine variants;
   make aggressive format standardization opt-in (matches PRODUCT.md's trust contract).

### 🎯 Real-data goalpost (fine-tuned model)
| metric | NO-OP | HEURISTIC | **target** |
|---|---|---|---|
| recovery | 0.975 | 0.874 | **≥ 0.985** (beat NO-OP by fixing typos) |
| repair_recall | 0.000 | 0.000 | **≥ 0.30** (fix real typos via clustering) |
| broken | 0 | 2021 | **≤ 50** (don't over-standardize) |

The model plugs into `_score(dirty, clean, model_output)` exactly like the heuristic.

> Data auto-fetched to `data/real/hospital/` (gitignored). Add Flights/Beers/CleanML the
> same way for breadth.
