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

**Model:** Qwen3-4B (≤4B → Tiny Titan), run locally via llama.cpp (GGUF).
**Targeting all six bonus quests.** See project memory for full guidelines.

## Develop

```bash
uv sync          # install deps
uv run app.py    # launch the Gradio app locally
```

## Deploy

Create a Space under `build-small-hackathon`, then push this repo to its
git remote. The YAML header above configures the Space.

## Submission checklist

- [ ] Model ≤ 32B params (≤ 4B unlocks Tiny Titan)
- [ ] Gradio app on a HF Space under `build-small-hackathon`
- [ ] Short demo video
- [ ] Social-media post
- [ ] (bonus) fine-tune published, llama.cpp runtime, custom `gr.Server` UI,
      agent trace shared, blog/field-notes post
