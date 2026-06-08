---
title: Hackaton Small
emoji: 🏔️
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
---

# ScrubData — hands-off data cleaning

Entry for the **Build Small Hackathon** (Gradio · Hugging Face), 🏡 Backyard AI track.

> **Upload your dirty spreadsheet. Get clean data back. No config.**
>
> For the office/ops person trying to do their job while their data is a mess.

## How it works

The small model is the **planner**, not a row-by-row workhorse:

1. **Profile** the data deterministically in pandas (dtypes, nulls, duplicates,
   whitespace, cardinality, sample values).
2. **Plan** — the model reads the profile + a sample and emits a structured
   cleaning plan: semantic column types, ID/date/categorical detection, and the
   fuzzy work rules can't do — canonicalizing messy categories
   (`USA` / `U.S.A` / `united states` → `United States`), anomalies, formats.
3. **Execute** the plan deterministically in pandas (reliable, fast, auditable).
4. **Narrate** a plain-English "what was wrong / what I fixed" report.

**Model:** `Qwen3-4B-Instruct-2507` (4.0B → Tiny Titan), fine-tuned via QLoRA, run via
llama.cpp (GGUF). The deterministic `scrubdata.executor` applies the plan.

## Run it

```bash
uv sync                                            # install deps
uv run server.py                                   # gr.Server + custom UI (heuristic planner)

# use the fine-tuned model as the planner (needs Ollama + the GGUF, see notebooks/Modelfile):
ollama pull hf.co/ricalanis/scrubdata-qwen3-4b-v4-q8:Q8_0
ollama create scrubdata-ft -f notebooks/Modelfile
SCRUBDATA_MODEL=scrubdata-ft uv run server.py      # model planner, heuristic fallback

uv run python -m scrubdata.cli messy.csv -o clean.csv --plan plan.json   # CLI
uv run pytest tests/                               # engine tests (18)
```

The planner is pluggable (`scrubdata/active.py`): set `SCRUBDATA_MODEL` to a local Ollama
model id to use the fine-tune (alias-level canonicalization), else it runs the deterministic
heuristic. Per column-batch it falls back to the heuristic if the model errors, so the app
never breaks.

## Repo map
- `scrubdata/` — the engine: `profiler` · `planner` (heuristic; swaps for the model) ·
  `executor` (deterministic) · `report` · `trace` (Open-trace) · `cli` · `model_planner`.
- `training/` — self-verified synthetic data generator (`build_dataset.py`) backed by real
  vocabularies; pushes to `ricalanis/scrubdata-sft`.
- `eval/` — harness with a **frozen gold set** (`gold.jsonl`): validity + op/canon F1 +
  executor recovery (`run_eval.py`), real OOD slice (`run_real.py`, Raha hospital),
  one-command model eval vs goalposts (`run_finetuned.py`). See `eval/README.md`.
- `notebooks/` — Colab QLoRA training (`train_qlora.py`, `train_colab.ipynb`) + model card.
- `frontend/` — custom `gr.Server` UI.

## How good is it
Measured on a frozen held-out gold set (heuristic = rule baseline; oracle = perfect plan):
the heuristic does the easy work (op_f1 0.96, real-typo repair_recall 0.29) but is ~blind
to alias-level canonicalization (canon_f1 0.13). The fine-tune's job is to close that —
goalpost **canon_f1 ≥ 0.85, recovery ≥ 0.95**. Full numbers in `eval/README.md`.

## Submission checklist
- [x] Model ≤ 4B (Tiny Titan) — `Qwen3-4B-Instruct-2507`
- [x] Custom `gr.Server` UI (Off-Brand) · agent traces (`scrubdata/trace.py`, Open-trace)
- [x] Fine-tune published (Well-Tuned) · llama.cpp GGUF (Llama Champion)
- [ ] Gradio app live on a HF Space under `build-small-hackathon`
- [ ] Short demo video · social post · field-notes write-up
