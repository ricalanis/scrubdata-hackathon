---
title: ScrubData
emoji: 🏔️
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.16.0
app_file: server.py
pinned: true
license: mit
---
# ScrubData — hands-off data cleaning, with the receipts

Entry for the **Build Small Hackathon** (Gradio · Hugging Face), 🏡 Backyard AI track.

> **Drop a messy export. Get clean data back — every change named, reversible, and
> explained. Anything sensitive is protected locally. The judgment calls stay yours.**
>
> For the office/ops person trying to do their job while their data is a mess.

**Live Space:** https://huggingface.co/spaces/build-small-hackathon/scrubdata

## How it works

A small local model is the **planner**, never a row-by-row editor:

1. **Profile** — pandas aggregates each column into a value–frequency distribution
   (scale-invariant: a million rows profile like a hundred).
2. **Plan** — the model reads the profile and emits a structured JSON cleaning plan:
   canonicalization mappings, format fixes, dedup, anomaly flags.
3. **Ground** — canonical forms are never invented: values reconcile against reference
   taxonomies (GeoNames 196k cities, ISO countries/states, and a pluggable **entity
   reference** built from harvested vocabularies — ToughTables/MusicBrainz/Wikidata/ROR,
   ~100k entities) with fuzzy retrieval; ambiguous matches **abstain** and surface for
   human review (calibrated: 90% precision at the default threshold, ≥95% at 0.91).
   Profiles carry **suspect_values** — rare anomalous surfaces with evidence-backed
   candidates — so high-cardinality columns are no longer invisible to the planner
   (measured: five all-unique-surface benchmark tables went 0.0 → 0.96 F1 at zero damage).
4. **Verify** — every model-proposed mapping is scored by deterministic evidence
   (errors-are-rare frequency gates, variant similarity, reference agreement); entries
   below the confidence threshold (`SCRUBDATA_TAU`, default 0.5) become review flags
   instead of edits. The shipped **verified union planner** (gated model plan ∪ grounded
   heuristic) measures **0.905 precision @ 0.413 coverage** on hospital's 509 real errors
   — the gated model plan alone is 0.993 @ 0.287.
5. **Protect** — PII is detected locally (Luhn/IBAN checksums + a 44M OpenMed-PII
   classifier): cards/SSNs masked format-preservingly, contacts flagged, **0/360 residual
   PII** after masking in our leak test.
6. **Execute** — deterministic pandas applies the plan. No silent edits, by construction;
   every run exports an audit trail (OpenTelemetry-GenAI spans + open traces).

**Model:** `Qwen3-4B-Instruct-2507` (Tiny Titan), QLoRA fine-tuned on **execution-verified**
synthetic + real-derived data (every training plan provably recovers the clean table),
runnable via llama.cpp GGUF.

## Measured (not vibes)

- **Canonicalization micro-F1 0.90 (best single run; 0.80 ± 0.01 over 3 training seeds)** for the 4B
  fine-tune vs **0.45** for a much larger generic model vs **0.15** for rules.
- Real errors (5-benchmark macro): grounded cleaning reaches REAL-F1 **0.225**, 3.9×
  OpenRefine kNN (0.058) and 5.7× fingerprint (0.039); the verified-union gate repairs
  41% of hospital's 509 real errors at **0.905 precision**, every declined merge
  surfaced for review.
- Evaluated on a **65-dataset suite** (Raha benchmarks + seeded error injection over 15
  open-data domains) with a churn-neutral metric that can't be gamed by mass rewriting.
- Full write-up: `docs/paper/` (preprint draft) · details in `eval/README.md`.

## Run it

```bash
uv sync
uv run server.py                                   # gr.Server + custom UI (grounded heuristic)

# fine-tuned model as planner (needs Ollama + the GGUF, see notebooks/Modelfile):
ollama pull hf.co/ricalanis/scrubdata-qwen3-4b-v6-q8:Q8_0
ollama create scrubdata-ft -f notebooks/Modelfile
SCRUBDATA_MODEL=scrubdata-ft uv run server.py      # model planner, heuristic fallback

SCRUBDATA_PII_NER=1 uv run server.py               # +44M NER for name/address columns
uv run python -m scrubdata.cli messy.csv -o clean.csv --plan plan.json
uv run pytest tests/                               # engine + scorer tests (68)
```

## Repo map
- `scrubdata/` — `profiler` · `planner` · `reconcile` (reference grounding + abstain) ·
  `grounded` (RACOON wrapper) · `verifier` (selective prediction + union planner) ·
  `pair_profile` (candidate-constrained canonicalization, opt-in) · `pii` (checksum +
  NER tiers, mask/hash/pseudonymize) · `executor` · `observability` · `trace` ·
  `baselines` (OpenRefine) · `cli`.
- `training/` — execution-verified synthetic generator + real-data derivation
  (`real_data.py`: paired benchmarks + frequency-derived unpaired open data).
- `eval/` — frozen gold · wide suite + double-macro north-star (`run_real_multi.py`) ·
  ablations · calibration (risk–coverage) · PII leak test.
- `docs/paper/` — preprint: *Verified Cleaning Plans: Plan-Level Selective Prediction
  Turns Local LLM Planners into Trustworthy Table Cleaners*.
- `scripts/` — Modal train/eval (headless GPU loop), trace publishing.

## Submission checklist
- [x] Model ≤ 4B (Tiny Titan) — `Qwen3-4B-Instruct-2507`
- [x] Custom `gr.Server` UI (Off-Brand) — warm, plain-English, audit-first
- [x] Fine-tune published (Well-Tuned) — `ricalanis/scrubdata-qwen3-4b` (+GGUF, Llama Champion)
- [x] Agent traces published (Open-trace) — `build-small-hackathon/scrubdata-traces`
- [x] Gradio app live on a HF Space under `build-small-hackathon`
- [ ] Short demo video · social post
