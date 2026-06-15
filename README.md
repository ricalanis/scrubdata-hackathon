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
tags:
  - track:backyard
  - sponsor:modal
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:llama
  - achievement:sharing
  - achievement:fieldnotes
---
# ScrubData — hands-off data cleaning, with the receipts

Entry for the **Build Small Hackathon** (Gradio · Hugging Face), 🏡 Backyard AI track.
Runs a ≤4B model — a local-runnable GGUF, no third-party AI APIs → also in the running for
**Tiny Titan**, **Off-Brand**, **Best Demo**, **Best Agent**, and **Bonus Quest Champion**
(all six quests claimed above).

<!-- SUBMISSION LINKS (all set for June 15):
  Demo video: https://www.loom.com/share/2fa868147527496e8097d82dd546d663  [DONE]
  Social post: https://x.com/ric_alanis/status/2066598533738692983  [DONE]
  These links + this write-up are required by the build-small-hackathon /submit tool. -->

> **Hosted demo vs. local — read this.** This Space is a **no-install demo** that cleans with
> the real **Qwen3-4B fine-tune** by default (served on an A100 GPU, ~1 min/clean warm; first
> run after idle ~2 min on cold start) — the whole point
> is the small model doing the work. Your file is processed on Hugging Face / the GPU endpoint
> (sent to no third-party API, not stored); untick the box for an instant deterministic pass.
> The **privacy story is a property of running it yourself**: `SCRUBDATA_MODEL=scrubdata-ft uv
> run server.py` reads and cleans your file on-device with the same fine-tune — nothing leaves
> your machine. The app labels its own mode honestly (the ribbon says which one you're using).
> Same auditable plan→verify→execute pipeline either way.

> **Modal** (`sponsor:modal`): the hosted Space cleans with the Qwen3-4B fine-tune served from a
> **scale-to-zero Modal GPU endpoint** (`scripts/modal_serve.py`, Ollama on an A100; $0 when idle,
> pre-warmed on page load to hide the cold start). Modal also drove the headless training +
> evaluation loop behind the published model. The deterministic planner is the silent fallback
> if the GPU is cold or down, so the demo never hard-fails.

> **Drop a messy export. Get clean data back — every change named, reversible, and
> explained. Anything sensitive is protected locally. The judgment calls stay yours.**
>
> For the office/ops person trying to do their job while their data is a mess.

**Built by:** [@ricalanis](https://huggingface.co/ricalanis) (solo) · 🤗 Hugging Face: `ricalanis`
**Live Space:** https://huggingface.co/spaces/build-small-hackathon/scrubdata
**Code (open source):** https://github.com/ricalanis/scrubdata-hackathon
**Demo video:** https://www.loom.com/share/2fa868147527496e8097d82dd546d663
**Write-up / post:** https://x.com/ric_alanis/status/2066598533738692983

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

## The app (what judges see)
A custom `gr.Server` frontend (no default Gradio chrome — the **Off-Brand** quest), built
around the trust story:
- **YOUR CALL cards** — when the model is genuinely torn (e.g. *Slovia → Slovakia 86% vs
  Slovenia 86%*) it abstains and hands you the tie with both candidates; pick the right one
  and **stage several decisions**, then "✓ Clean now" replays them as one plan.
- **Named, reversible receipts** — every edit shows as a row in the audit grid with its op +
  rationale and a before/after diff; nothing is silent.
- **PII review cards** — embedded cards/SSNs (Luhn/strict-regex) flagged and masked
  format-preservingly, on-device.
- **Save / replay recipe** — export the cleaning plan as JSON and re-apply it to next week's
  export in one click (the "Monday ritual").
- **Honest, self-aware copy** — the app injects its own runtime state and the ribbon says
  exactly which planner ran and where your data was processed.
- **A fun, size-aware ETA timer** + cold-start readiness gate + page-load GPU pre-warm, so
  the model path feels responsive and never lies about progress.
- Drag-and-drop, two bundled sample exports, mobile-responsive layout.

## What real users told us (and what we changed)

Before submission we put the live Space in front of people who **aren't** data folks — the
exact audience the tool is for — and sent the link with one line: *"if you have a messy
spreadsheet, try it."* The most useful finding wasn't a bug. It was that the word
**"cleaning" didn't land**:

- One tester read "clean my Excel" as *deleting* data:
  *"¿Te refieres a que elimine algo de algún archivo?"* — "You mean it removes something
  from the file?"
- Another didn't know where to begin:
  *"¿eso del Excel te lo subimos ahí o cómo?"* — "the Excel thing, do we upload it there,
  or how?"
- The clearest explanation in the whole thread was one we had to type by hand in chat:
  *"it fixes text errors — names, phones, emails, cities."* That sentence wasn't anywhere
  in the product.

So we changed the product to **show** what cleaning means instead of naming it:

- the hero now leads with a literal before→after strip
  (`nigeia → Nigeria`, `Calfornia → California`, `Ana@GMAIL.com → ana@gmail.com`,
  `415.555.0192 → (415) 555-0192`) so the value is obvious *before* any upload;
- the headline is the sentence that worked in chat — **"Fix the messy text in your
  spreadsheet"** — and the copy says plainly **"I never delete your data"** (killing the
  "does it erase things?" misread);
- a one-click **"watch it run on a sample file"** path removes the "where do I start?" wall;
- jargon labels are gone ("HR payroll (with PII)" → "an HR file with sensitive data").

n is small and informal (friends-and-network, ~3 people), so this isn't a usability *study* —
but the feedback was real, it pointed at a failure of the *framing* rather than the engine,
and it changed the build. The persona "Maria" below is the controlled walk-through; the
quotes above are verbatim from people we know.

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
SCRUBDATA_MODEL=scrubdata-ft uv run server.py      # model planner, heuristic fallback (on-device)

SCRUBDATA_PII_NER=1 uv run server.py               # +44M NER for name/address columns
uv run python -m scrubdata.cli messy.csv -o clean.csv --plan plan.json
uv run pytest tests/                               # engine + scorer tests (69)
```

The hosted Space serves the same fine-tune from a scale-to-zero **Modal A100**
(`scripts/modal_serve.py`) and the planner adds `format=json` on that path
(`SCRUBDATA_OLLAMA_FORMAT_JSON=1`) to grammar-constrain the GGUF on the A100's kernels.
`scripts/modal_warm.py on|off` pins/un-pins a warm container (no cold start) without a
redeploy — leave it `off` (scale-to-zero, $0 idle), flip `on` for a live judging window.

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

## Research & resources
Everything behind the demo is public:
- 🚀 **Live Space** — https://huggingface.co/spaces/build-small-hackathon/scrubdata
- 💻 **Code (open source)** — https://github.com/ricalanis/scrubdata-hackathon
- 🧠 **Fine-tuned model** — https://huggingface.co/ricalanis/scrubdata-qwen3-4b
  (Q8_0 GGUF: https://huggingface.co/ricalanis/scrubdata-qwen3-4b-v6-q8)
- 📊 **WildClean dataset** (real-world dirty tables + injected-error benches) —
  https://huggingface.co/datasets/ricalanis/wildclean
- 🔍 **Agent traces** (OpenTelemetry-GenAI spans from real runs) —
  https://huggingface.co/datasets/build-small-hackathon/scrubdata-traces
- 📄 **Preprint** — *Verified Cleaning Plans: Plan-Level Selective Prediction Turns Local
  LLM Planners into Trustworthy Table Cleaners* (`docs/paper/main.pdf`)
- 📓 **Field notes** (the build story, failures included) — `docs/FIELD_NOTES.md`
- 🛠️ **Tool reference** (the whole system, end to end) — `docs/TOOL_REFERENCE.md`

## Built with Codex
The final review-and-refine pass used **OpenAI Codex** (gpt-5.5) as a reviewer / last
refiner — not to write the product, but to harden it. It added the executor's
never-corrupt-clean-data regression tests, made column sanitization collision-proof,
did the accessibility pass (ARIA + keyboard + reduced-motion + focus-visible), and wrote
characterization tests for the reference matcher. Every change was human-reviewed and
verified green (84 tests, golden behavior unchanged) before commit; the commits are
attributed to `@codex` in the git history.

## Submission checklist (verified against the build-small-hackathon `/submit` tool)
- [x] Public Gradio Space in the `build-small-hackathon` org
- [x] Every model ≤ 32B (here ≤ 4B → **Tiny Titan**-eligible): `Qwen3-4B-Instruct-2507`
- [x] README `tags:` set — `track:backyard` + all six `achievement:*` quests (above)
- [x] **Off the Grid** (`offgrid`) — no third-party AI APIs; the planner is a local-runnable GGUF (Qwen3-4B). Self-hosted = fully on-device (zero external egress); the hosted demo serves the *same* model from a self-managed Modal GPU, not a SaaS API
- [x] **Well-Tuned** (`welltuned`) — fine-tune published: `ricalanis/scrubdata-qwen3-4b` (+ `-v6-q8` GGUF)
- [x] **Off-Brand** (`offbrand`) — custom `gr.Server` HTML/CSS frontend, not default Gradio
- [x] **Llama Champion** (`llama`) — runs through llama.cpp (Q8_0 GGUF)
- [x] **Sharing is Caring** (`sharing`) — agent traces on the Hub: `build-small-hackathon/scrubdata-traces`
- [x] **Field Notes** (`fieldnotes`) — build report: `docs/FIELD_NOTES.md`
- [x] Write-up in this README (idea + tech)
- [x] **Demo video** link in README: https://www.loom.com/share/2fa868147527496e8097d82dd546d663
- [x] **Social post** link in README: https://x.com/ric_alanis/status/2066598533738692983
- [x] Confirm deadline time/timezone — **June 15 2026, 23:59 UTC** (confirmed on the hackathon page)

Judged (no tag needed, just qualify): Tiny Titan · Off-Brand prize · Best Demo · Best Agent · Bonus Quest Champion.
