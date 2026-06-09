# ScrubData — paper scaffold & related-work map

**Working title:** *Small fine-tuned planners with execution-verified data and calibrated
abstention match larger models on tabular canonicalization.*

**One-line claim (measured):** a ≤4B fine-tune that emits a *cleaning plan* (not edited cells)
reaches `canon_f1 0.86` on alias-level canonicalization vs `0.45` for a large generic model and
`0.13` for a rule heuristic — and, with reference grounding + calibrated abstention, beats the
tool people actually use (OpenRefine) on a wide validation suite at far lower damage.

## Contributions (the combination is the novelty — not "LLM cleans data")
1. **Planner/executor decomposition.** The model proposes a structured JSON plan; deterministic
   pandas executes it. Auditable, reversible, **no silent edits** (`observability.py`,
   `trace.py`). This is the trust/monitorability contract.
2. **Execution-self-verified synthetic SFT.** Every training example's plan is checked to
   actually recover the known-clean original by *running the executor* (`training/build_dataset.py`).
   A clean, citable data-generation method (drops non-recovering examples).
3. **Reference grounding + calibrated abstention.** Canonicalization is reconciled against a
   type-scoped taxonomy (GeoNames/pycountry; `reconcile.py`, `grounded.py`); the system ABSTAINS
   under ambiguity instead of hallucinating a canonical (`eval/calibration.py`: risk-coverage +
   ECE). Structural fix for the over-correction larger models also exhibit.
4. **Aggregation + column-batching.** Prompt size scales with *distinct values*, not rows
   (`profiler.py` value_counts + `model_planner.make_batched_planner`).

## Related work (position against — reviewers know this field)
- **Error detection/repair:** Raha & Baran (Mahdavi et al.), HoloClean (Rekatsinas et al. 2017,
  `arXiv 1702.00820`), GARF — we *use* their hospital/beers/flights/rayyan as OOD eval and cite
  GARF as the frequency-only baseline our grounding beats (it cannot supply a canonical for a lone
  column).
- **LLMs for data wrangling:** "Can Foundation Models Wrangle Your Data?" (Narayan et al. 2022),
  Jellyfish, Table-GPT/TableLlama (`2311.09206`), RetClean (`2303.16909`). We differ by being a
  *small fine-tuned planner* + grounding + abstain, not a large zero-shot value-editor.
- **Grounding / entity disambiguation:** RACOON (`2409.14556`), TURL (`2006.14806`), Belotti et al.
  table-EL (`2408.06423`), MTab — motivate retrieval-then-abstain and warn against memorizing
  canonicals into weights (TURL ~40% OOD collapse). See `taxonomy-grounding.md`.
- **The tool we beat:** **OpenRefine** clustering — fingerprint (key collision) + nearest-neighbor
  (kNN/edit-distance), reimplemented as `scrubdata/baselines.py` for head-to-head.
- **Selective prediction:** calibrated abstention / risk-coverage (El-Yaniv & Wiener; Geifman &
  El-Yaniv) — our ECE/AURC study; also the AI-safety monitorability framing.

## Experiments
- **Headline:** canon_f1 vs large-generic vs heuristic on frozen synthetic gold (Layer 1).
- **Wide north-star (`eval/run_real_multi.py`):** double-macro (error-type × domain) F1 + damage +
  abstain over Raha real-error sets **+ seeded error-injection** on 20+ harvested gov/GitHub clean
  domains (`eval/inject.py`); multi-seed 95% CIs. Hospital is 1 dataset of many.
- **Money result:** grounded vs OpenRefine fingerprint & kNN on the same suite (grounded wins F1 +
  damage; kNN over-merges — higher recall, low precision, high damage).
- **Calibration (`eval/calibration.py`):** risk-coverage, AURC, ECE; operating point for ≥95%
  precision via the abstain threshold.
- **Ablations to add:** −grounding, −abstain, −execution-verification, −aggregation.

## Honest limitations (the integrity reviewers reward)
- Reference *coverage* is the recall ceiling (Belotti) — uncovered entities abstain by design.
- Convention vs error: standardization (date→ISO, `%`→fraction) is product value, not damage —
  the metric is case/whitespace-normalized but a format-aware variant is future work.
- ECE shows mild over-confidence (difflib-ratio scores) — temperature/Platt scaling is future work.
- Some benchmark sources gated (CleanML/TableEG behind Dropbox/Drive; licenses noted).

## To-do before submission
multi-seed CIs (running) · −ablations · OpenRefine table with CIs · cs.AI endorser · selective-
prediction figure · keep the eval README's convention-vs-error honesty.
