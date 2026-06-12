# ScrubData — The Profound Tool Reference

> The single local document that explains the whole system: what it is, why every
> piece exists, where every number comes from, and what we learned building it.
> Written at the close of the research domain (2026-06-12). The paper
> (`docs/paper/main.tex`) is the citable account; THIS file is the operational one.

---

## 1. What ScrubData is

ScrubData is a **zero-config, zero-label, local** tabular data-cleaning system built
around one architectural commitment: **the model never touches data**.

A profiler aggregates each column into a bounded value-frequency profile; a small
(≤4B, locally-run) fine-tuned planner *proposes* a JSON cleaning plan; a
deterministic pandas executor *applies* it. The plan is the complete, inspectable,
reversible specification of every change. Three consequences define the product:

1. **No silent edits by construction** — every changed cell traces to a named,
   logged operation (verified at scale: 0 silent edits across 35 wild tables and a
   239-table GitTables trust audit).
2. **Abstention is first-class** — anything below confidence becomes a review flag
   ("YOUR CALL" card in the UI), never a quiet skip and never a guess.
3. **Profile-not-rows scaling** — the prompt scales with *distinct values*, not
   rows; a million-row table profiles like a hundred-row one, and no cell values
   leave the machine.

### The central finding (load-bearing, repeatedly measured)

**Model weights contribute approximately nothing to never-seen-table
generalization in this protocol class.** Five SFT retrains (v7–v10 + mixes, 109k
harvested real alias pairs) and a three-arm GRPO pilot (executor as verifiable
reward, including a random-reward control that reproduced the same format drift)
all failed to move held-out generalization. Every measured gain came from
**deterministic machinery gated by the plan-level verifier** (§5). Corroborated
independently by Spreadsheet-RL, arXiv:2601.05009, and arXiv:2606.02866.
Practical corollary: *to improve ScrubData, write a deterministic capability and
gate it with the verifier; do not collect more training data.*

---

## 2. The shipped pipeline (`scrubdata/active.py::get_planner`)

```
                       ┌──────────────────────────────────────────────┐
 df ──► profiler ──►   │  model path (only if SCRUBDATA_MODEL is set) │
        (bounded       │  batched (4 cols/call) local Ollama planner  │
        profile incl.  │  → per-batch fallback to heuristic on error  │
        suspects)      │  → grounded (reference taxonomies, RACOON)   │
                       │  → verify_plan(tau=SCRUBDATA_TAU, def 0.5)   │
                       └───────────────┬──────────────────────────────┘
                                       │  union_plans (model wins per surface;
                                       │  inherits deterministic ops + table ops)
        heuristic mock_plan ───────────┘
                                       ▼
                        executor.apply_plan → (clean_df, change_log)
                                       ▼
                 report.render_report · trace.log_run · observability
```

- **No model configured** → `mock_plan` (grounded deterministic heuristic) alone.
  The app always produces a plan; the model is an upgrade, never a dependency.
- **Measured operating point** (hospital, 509 real errors): union **0.905
  precision @ 0.413 coverage**; gated model alone 0.993 @ 0.287; 3-seed
  0.891±0.012 @ 0.396±0.025. Precision flat 0.89–0.91 for τ∈[0.2,0.8].

Entry points: `uv run server.py` (FastAPI + UI), `app.py` (HF Space/Gradio),
`scrubdata/cli.py` (`scrubdata <file.csv> -o out.csv --report r.md --plan p.json`).

### Environment variables

| Var | Default | Meaning |
|---|---|---|
| `SCRUBDATA_MODEL` | unset | local Ollama model id (e.g. `scrubdata-ft-v6`); unset = heuristic only |
| `SCRUBDATA_TAU` | `0.5` | per-entry verifier threshold on model mappings |
| `SCRUBDATA_HC_TAU` | `0.8` | stricter bar for heuristic suspect-mappings (no model cross-check there) |
| `SCRUBDATA_PAIR_PROFILES` | off | WS2 candidate-constrained planning (measured redundant with verifier; off by default) |
| `SCRUBDATA_PII_NER` | off | OpenMed-PII 44M NER tier on top of deterministic validators |

---

## 3. Module map (`scrubdata/`)

| Module | Role | Key facts |
|---|---|---|
| `profiler.py` | column → bounded profile | `VALUE_COUNTS_CAP=80` (high-card cols: top-8 only) + `suspect_values` section (the visibility fix); `truncated_values` count keeps honesty about what's hidden |
| `detect.py` | typing + issue predicates | `detect_semantic_type` (zip/ZCTA/Excel-serial guards), `date_formats_consistent` (collapses digit AND alpha runs; 90% dominant-shape), `percent_formats_consistent` (90%), `has_mojibake`, `is_missing` |
| `planner.py` | deterministic heuristic planner | `mock_plan`, `_column_operations`, `_suspect_canonicalize` (τ_hc=0.8), `detect_entity_groups` (cross-row voting detection), emits `fix_encoding` BEFORE `strip_whitespace` (order-critical), `off_convention_dates` visible-abstention flags |
| `executor.py` | the only thing that touches cells | op dispatch (§4); unknown ops are no-ops (forward-compatible); returns `(df, change_log)`; `resolve_by_majority` table op lives here |
| `verifier.py` | WS1 selective prediction | `entry_confidence` (3 hard gates, §5.0), `verify_plan` (also enforces convention gates on MODEL-emitted parse_date/parse_percent — the model path otherwise bypasses them), `union_plans` (order-preserving op inheritance via `reversed(inherit)`) |
| `reconcile.py` | reference grounding | `ReferenceIndex`, `default_index()` loads toughtables_ref (contamination-guarded: excludes the 8 benchmark tables) + MusicBrainz hints + Wikidata companies + ROR; `infer_reference_type` needs **≥20% exact entity hits** (over-fire guard); falls back to `training/harvests/` for Space/clone parity |
| `grounded.py` | RACOON wrapper | model never free-generates a canonical for a reference-typed column |
| `pair_profile.py` | suspects + WS2 candidates | `suspects_for_column` (≤25/col, bounded: 4k rare cap + cheap prefilters before SequenceMatcher — 40min→24s fix), `candidate_pairs`, `constrain_plan` |
| `model_planner.py` | Ollama backends | `make_local_ollama_planner`, `make_batched_planner(batch_size=4)`, JSON extraction |
| `prompt.py` | prompt/training contract | `_profile_for_prompt` (compact suspects), `build_chat_example` (training-data side of the same contract — change one, regenerate the other) |
| `pii.py` | PII second task | deterministic validators (Luhn, IBAN, phone) + allowlist + coverage vote; optional 44M NER; `mask/hash/pseudonymize` |
| `active.py` | THE composition | `get_planner()` — §2 |
| `cli.py` / `report.py` / `trace.py` / `observability.py` | UX + audit | CLI, markdown report, JSONL traces, monitor summary/OTel span |
| `baselines.py` | OpenRefine kNN/fingerprint reimplementations | the zero-config comparison class |
| `refdata/cities.txt` | seed gazetteer | plus everything in `training/harvests/*.jsonl` |

---

## 4. Operation vocabulary (the executor's closed set)

**Column ops** (`_apply_column_op`): `strip_whitespace`, `normalize_punctuation`,
`fix_encoding` (lossless cp1252/latin-1↔utf8 round-trip, mojibake-marker-reduction
gated), `normalize_disguised_nulls`, `parse_currency`, `parse_number`,
`parse_percent` (abstains on bare values — no /100 corruption),
`parse_date`, `standardize_boolean`, `standardize_phone` (7-digit → `DDD-DDDD`),
`normalize_email`, `standardize_case`, `canonicalize_categories` (mapping-driven;
the verifier's subject), `flag_pii` (log-only), `mask_pii`, `hash_pii`,
`pseudonymize_pii`. Unknown op → no-op.

**Table ops**: `drop_empty_columns`, `drop_empty_rows`, `drop_exact_duplicates`,
`resolve_by_majority` (§5.3).

Op-order invariant: **`fix_encoding` must precede whitespace/punctuation ops** —
they destroy the UTF-8 byte patterns repair needs (grader-reproduced bug; fixed in
both heuristic emission and union inheritance).

---

## 5. The five deterministic capabilities (what actually generalizes)

### 5.0 Plan-level verifier (WS1) — `verifier.entry_confidence`
Every non-grounded `canonicalize_categories` entry `raw→canon` is scored with
three HARD gates, each killing a measured hospital failure class:
- **errors are rare**: `freq(raw) ≥ 3` → 0.0 (frequent = legit data; "de kalb"×92)
- **repair to dominance only**: `freq(canon) < max(2, 2·freq(raw))` → 0.0
  ("yex→yexu", typo mapped to a worse typo)
- **code discipline**: digit-bearing values repair only if letter-part similarity
  ≥0.85 AND digits identical (allows `amix-2→ami-2`, blocks `ak_→al_`)
Survivors score `sim × (0.5 + 0.5·support)`; below-τ entries become review flags.

### 5.1 Suspect surfacing (visibility) — `pair_profile.suspects_for_column`
The 80-value profile cap structurally hides high-cardinality dirty cells from ANY
planner (proved by the v8/v9 retrains: more data couldn't fix what the model
couldn't see). Every text-ish column profile now carries ≤25 `suspect_values`:
rare surfaces + evidence-backed candidates (frequency dominance, edit similarity,
reference membership). The heuristic maps suspects clearing `entry_confidence ≥
SCRUBDATA_HC_TAU=0.8`; the rest become flags.

### 5.2 Generic entity reference — `reconcile.default_index`
Open vocabularies (ToughTables-derived ref [8 bench tables excluded], MusicBrainz
search-hint misspellings, RxNorm, Wikidata companies, ROR, GeoNames, OpenFlights,
O*NET, nicknames) as a pluggable reference type. Typing requires **≥20% exact
hits** of distinct values (fuzzy coverage alone over-fires on name-like columns —
measured). Cracked the all-unique regime: 5 ToughTables tables **0 → 0.955–0.957
F1 at 0.0000 damage** (~62k corrections) — where no in-column frequency signal
exists at all.

### 5.3 Cross-row majority voting — `planner.detect_entity_groups` + `resolve_by_majority`
Tables repeating a real-world entity across rows (flights reported by many
sources) carry their own repair signal. Detection: compact-token key columns,
median multiplicity 3–30, ≥2 votable string columns with majority-bearing
disagreement + ≥2 distinct majorities, date-share ≤0.3 guard. Execution: resolve
thin dissenting minorities to group majority; skips missing-like keys;
min_share/min_group clamped. **False-consensus guard**: mean minority share ≥0.25
→ decline (legitimate correlated updates, not reporting errors — a flat volume cap
was measured to destroy the legitimate regime and replaced). Measured: flights
heuristic 0.044→**0.164** F1; hospital heuristic 0.092→**0.186**.

### 5.4 Convention conservatism — `detect.*_formats_consistent` + `verify_plan`
Never re-format an internally consistent column: date/percent ops gated on
dominant-shape inconsistency (digit+alpha runs collapsed, 90% rule); zip/postal
names never typed phone/date; Excel-serial typing needs a date-suggestive name.
Suppressed minorities surface as `off_convention_dates` flags. The verifier
enforces the same gates on model plans at the verification boundary (the model
path otherwise bypasses heuristic emission gates entirely).

---

## 6. Evaluation (how every number regenerates)

One scoring contract — `eval/run_real_multi.py::score()` — **churn-neutral,
convention-tolerant**: sem-equal = numeric-tolerant OR strip+casefold equal; pure
case/whitespace churn counts as nothing; a fix requires acting; **damage** =
clean cells corrupted / clean cells; **silent edits** = changed columns minus
log-attributed columns (must be 0).

| Harness | Command | What it measures | Current numbers |
|---|---|---|---|
| Money table | `python -m eval.run_real_multi` | 65-set suite, 3 seeds | grounded NORTH 0.203±0.003; REAL-F1 0.174 vs OR-kNN 0.058 |
| WS1 gate | `python -m eval.precision_curve --plan eval/results/v6_hospital_raw_plan.json --union` | precision–coverage curve | **0.905 @ 0.413** (τ=0.5) |
| Paired bench | `python -m eval.paired_bench` | 42 dirty/gold pairs | unseen-35 macro F1 **0.363** @ dmg **0.0219** |
| Wild bench | `python -m eval.wild_bench` | 35 uncurated tables, behavioral + inject-recovery | recovery 0.207; **0 silent edits** |
| Trust audit | `python -m eval.gittables_audit` | 239 GitTables clean-lake | 239/239 valid, 0 crashes, 0 silent edits |
| Generalization | `python -m eval.generalization` | held-out-source (train: hospital/beers/movies_1 · eval: flights/rayyan/ed2) | GEN-F1 0.058, VR 0.108, dmg 0.036 |
| RADAR board | `python -m eval.radar_bench` | regime boundaries by artifact type | abstains on missingness ✓; reasoning-class = frontier territory |
| Baselines | `eval/run_baran.py`, `modal run scripts/modal_jellyfish.py` | disclosed-protocol comparisons | Baran (oracle+20 labels) 0.811; Jellyfish-13B 0.074 |
| Calibration / PII | `eval.calibration`, `eval.pii_leak` | abstention quality / leak test | AURC 0.120, ECE 0.169; 0/360 residual PII |

**Eval-source discipline**: TRAIN_SOURCES["v6"]={hospital,beers,movies_1};
EVAL_SOURCES={flights,rayyan,ed2_restaurants}. Never crossed.

---

## 7. Model & artifacts

| Artifact | Where | Notes |
|---|---|---|
| Champion adapter | Modal volume `scrubdata-v5-adapter` `/v5_seed21` (= "v6") | survived v7–v10 challenges + GRPO |
| Merged model | `hf.co/ricalanis/scrubdata-qwen3-4b` | card carries the v2 finding |
| Q8 GGUF | `hf.co/ricalanis/scrubdata-qwen3-4b-v6-q8` | **Q8_0 only — Q4_K_M corrupts** (Unsloth 2026.6.x); non-thinking Modelfile required (`notebooks/Modelfile`); suppress tokens 151657/151658 under transformers |
| Benchmark | `hf.co/datasets/ricalanis/wildclean` | 33 redistributable pairs + loaders.py for 9 license-gated + gittables250 + 10 vocabs + frozen results; first cleaning bench with damage + silent-edit accounting |
| Demo | `hf.co/spaces/build-small-hackathon/scrubdata` | deploy = `HfApi.upload_folder` of `git archive HEAD` — **NO GitHub auto-sync** |
| Paper | `docs/paper/main.tex` + `numbers.tex` | compile: `~/.local/bin/tectonic main.tex` (no pdflatex on this machine) |
| Vocabs | `training/harvests/*.jsonl` (15MB, 13 files) | loader falls back here for clone parity |

Modal patterns: `--detach` for anything long; results land in Modal Dicts
(`scrubdata-train-results`, `scrubdata-eval-v5-results`, `scrubdata-suite-results`).
**Budget status at domain close: ~$187 of $212 ceiling — Modal HALTED.**

---

## 8. Negative results ledger (measured, do not re-litigate)

1. **v7–v10 SFT retrains**: 109k harvested alias pairs, episode mixes, suspects
   contract — GEN flat/worse. Mixing harvested pairs **dilutes** executor-verified
   synthetic skill (monotonic dilution law across mix ratios; mixH 0.677).
2. **GRPO pilot, 3 arms** (main, KL-anchored v2, random-reward control): all
   degrade format at 4B/LoRA/$30 scale; the control proved the drift is an RL
   artifact (cf. "Spurious Rewards"). Published RLVR wins used real infra
   (verl, 4×H100×40h). Episodes corpus (600, `training/build_grpo_episodes.py`) +
   hand-rolled loop (`scripts/modal_grpo.py`) committed for a future attempt.
3. **Uniform verification of existing low-card mappings** (A1 per-class
   thresholds): 0.905→0.890 — reverted.
4. **Strict entity-typing thresholds** (0.90/0.05): cost more than bought — reverted.
5. **WS2 candidate constraining composed with verifier**: 0.876 @ 0.387 < union at
   same τ — redundant gating of the same failure class; available, off by default.
6. **Flat volume cap on cross-row voting**: destroyed the legitimate
   dense-disagreement regime — replaced by the false-consensus guard.
7. **Frozen-gold synthetic yardstick predates the suspects prompt contract** —
   regenerate gold before ever quoting synthetic canon_f1 again.

## 9. Known-open (graded non-blocking)

`_parse_date` per-value dayfirst; i18n name guards; mojibake fixpoint /
sequence-plausibility; backlog sources: CMS API, NHTSA, Canada contracts, Matelda
~6,670 pairs, GLEIF/USDA vocabs, WDVC-16. Reasoning-class artifacts (RADAR) are
explicitly out of protocol class — frontier-model territory.

## 10. Where deeper detail lives

`docs/PRODUCT.md` (trust contract) · `docs/SOTA.md` + `docs/ROADMAP_SOTA2.md`
(position + research map) · `docs/CAPABILITY_GRADES.md` (12-agent adversarial
grading + must-fix ledger) · `docs/WILD_BENCH.md` / `docs/PAIRED_BENCH.md` /
`docs/GITTABLES_AUDIT.md` / `docs/DATASETS.md` (per-bench detail + licenses) ·
`docs/NIGHT_LOG.md` (stage-3 timeline) · `project-memory/` (agent memory snapshot).
