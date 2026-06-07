---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
datasets:
  - ricalanis/scrubdata-sft
tags:
  - data-cleaning
  - structured-output
  - json
  - tabular
  - gguf
  - llama.cpp
pipeline_tag: text-generation
---

# ScrubData Planner — Qwen3-4B (QLoRA)

A ≤4B model fine-tuned to be a **hands-off tabular data-cleaning planner**: it reads a
profile of a messy spreadsheet (per-column dtype, null/duplicate counts, detected
semantic type, sample values) and emits a **structured JSON cleaning plan**. Deterministic
pandas executes the plan — the model only *plans*. Built for the Build Small Hackathon
(Backyard AI track), targeting **Tiny Titan** (≤4B) and **Well-Tuned**.

- **Base:** `Qwen/Qwen3-4B-Instruct-2507` (4.0B, Apache-2.0)
- **Method:** QLoRA (Unsloth), r=32, 2 epochs, on an A100
- **Data:** `ricalanis/scrubdata-sft` — self-verified synthetic pairs (every example's
  plan was checked to recover the known-clean original by running the executor) backed by
  real vocabularies (countries/states/currencies/cities/industries/units) for genuine
  canonicalization, plus anomaly-flag and typo-cluster cases.
- **GGUF:** `ricalanis/scrubdata-qwen3-4b-gguf` (Q4_K_M, llama.cpp).

## What it's for
Standardizing formats (dates/numbers/phones), canonicalizing inconsistent categories
(`USA`/`U.S.A`/`united states` → `United States`), normalizing disguised nulls,
de-duplicating, and flagging anomalies — with every change explained and reversible.

## Evaluation
Scored on a frozen held-out gold set + a real OOD slice (Raha `hospital`). The fine-tune
target is to clearly beat the rule-based heuristic, especially on **alias-level
canonicalization** (the fuzzy skill rules can't do).

| metric (synthetic) | heuristic | **this model** | oracle |
|---|---|---|---|
| op_f1 | 0.961 | _TBD_ | 1.000 |
| canon_f1 | 0.133 | _TBD_ | 1.000 |
| recovery | 0.627 | _TBD_ | 1.000 |
| real repair_recall | 0.293 | _TBD_ | — |

_(Numbers filled in from `eval/run_finetuned.py` after training.)_

## Usage (llama.cpp / Ollama)
```bash
ollama run hf.co/ricalanis/scrubdata-qwen3-4b-gguf
```
System prompt + profile→plan format: see `scrubdata/prompt.py` in the project repo.

## Limitations
Plans only — it never edits data directly. Format standardization is opinionated (parses
`100%`→`1.0`, reformats phones); on datasets with different conventions this is a feature,
not error-correction. Open-ended typo/entity-resolution beyond seen vocabulary is the
remaining hard tail.
