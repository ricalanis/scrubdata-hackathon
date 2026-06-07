# Training-data generation

Self-verified synthetic SFT data for the ScrubData planner: `(dirty profile → JSON
cleaning plan)` pairs. We generate **clean** tables, inject controlled mess, and
because *we* created the mess the ground-truth plan is known. Every example is then
**verified by running `scrubdata.executor`** (dirty + plan must recover the clean
original) — only perfectly-recovered examples are kept.

## Run

```bash
uv run training/build_dataset.py --n 2000 --out data/train.jsonl --seed 0
```

Output is chat-format JSONL (`messages`: system/user/assistant) using the shared
`scrubdata/prompt.py` serialization, so **training === inference**.

## Why it's hard (not just heuristic imitation)

The early version drew from toy pools (4 countries, 3 category sets) — the target
plans were exactly what the deterministic `mock_planner` already produces, so a
fine-tune would just clone a free heuristic. The generator is now backed by **real
vocabularies** the heuristic has no knowledge of:

- **`vocab.py`** — countries / US states / currencies (offline via `pycountry`),
  ~460 cities (curated aliases + `cities.txt`, open world-cities data), departments,
  job titles, status sets. Each canonical has realistic surface variants (aliases,
  ISO codes, casing, punctuation, single-char typos).
- Columns stay **low-cardinality** (a few canonicals each) so every dirty surface
  appears in the profile sample the model sees — the task is learnable *and* the
  executor can recover it.

Latest 2000-example build: **844 distinct canonical targets**, **33k surface→canonical
pairs**, all 10 semantic types, plus **anomaly flag-only** examples (teach surfacing
implausible values without changing them). ~93% of attempts verify; the rest are
dropped (the quality gate).

## Files
- `fields.py` — field archetypes (clean generator + matched corruptor).
- `vocab.py` — real vocabularies + the surface-corruption engine.
- `cities.txt` — 400 extra cities derived from open world-cities data.
- `generate.py` — assembles one example (columns + table corruptions + anomaly flags).
- `build_dataset.py` — loops, **verifies via the executor**, writes JSONL.
