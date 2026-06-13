# maria_crm_export.csv — demo cheat-sheet & planted-issue inventory

**Persona.** Maria, an ops coordinator, exported the company CRM on Monday morning.
383 rows × 12 columns. Engineered against the *shipped* pipeline
(`scrubdata.planner.mock_plan` + `reconcile.grounded_mapping` + `pii.detect_column_pii`),
regenerate with `uv run python samples/make_maria_crm.py` (deterministic, seed 20260612).

Columns: `company, contact_name, email, phone, country, state, signup_date, plan_tier,
mrr, status, cc_on_file, notes`.

## What ScrubData should do, by column

| Column | Planted mess | Expected action |
|---|---|---|
| **company** | mojibake (`MÃ¼ller GmbH`); case/whitespace variants of Acme; `Globex` vs `Globex Corp` | `fix_encoding` → `strip_whitespace` → `canonicalize_categories` folds `acme corp`/`ACME Corp`→`Acme Corp`. **Does NOT merge `ACME Corporation`** (see YOUR CALL #1). |
| **contact_name** | ALLCAPS / lowercase / leading-trailing spaces | `fix_encoding`, `strip_whitespace`, case-fold of exact case-variants. Rare names → **`suspect_values` review flag** (won't merge distinct people — abstention-first). |
| **email** | UPPERCASE + whitespace | `flag_pii [email]` (flag only, not masked — users need it) → `strip_whitespace` → `normalize_email`. |
| **phone** | 5 formats: `(555) x-y`, `555.x.y`, `+1 555 x y`, `555xy`, `1-555-x-y` | `flag_pii [phone]` → `standardize_phone`. |
| **country** | 4+ USA spellings + rare typos | `canonicalize_categories`: `U.S.A`/`us`/`USA.`→United States, `germny`→Germany, `Nigeia`→Nigeria, `Polnd`/`Swedn`→Poland/Sweden. **Abstains on the YOUR CALL items.** |
| **state** | abbrev chaos `CA`/`California`, typos | `canonicalize_categories`: `Calfornia`→California, `Wahsington`→Washington, `Virgina`→Virginia, `Mississipi`→Mississippi (cast to column's dominant case = ALLCAPS abbrevs). |
| **signup_date** | 3 formats: ISO, `M/D/YY`, `Month D YYYY`, `DD-MM-YYYY` | `parse_date` → ISO `YYYY-MM-DD`. |
| **plan_tier** | `Premium/premium/PREMIUM/Prem`, `Basic/…`, `Ent`, `Free` | `canonicalize_categories` case-folds the variants. |
| **mrr** | `$1,200.00` / `1200.00` / `1,200 USD` / `1200` | `parse_currency` → numeric. |
| **status** | `Active/active/ACTIVE`, `Churned`, `Trial`, `Lapsed` | case-fold canonicalization. |
| **cc_on_file** | 8 **Luhn-valid** fake cards, 4 surface formats | `flag_pii [credit_card]` (checksum-confirmed) → **`mask_pii`** → `****-****-****-5559` (last 4 kept). |
| **notes** | clearly-fake SSNs (`123-45-6789`, `078-05-1120`) in free text | Sparse free-text → column PII typer does NOT fire (needs ≥60% of distinct cells to match). SSNs survive as text — honest limitation; tier-2 NER would be needed to mask in-prose PII. |

## THE YOUR CALL moments (the abstention wow — these SHOULD surface, not auto-merge)

1. **Entity near-tie — `Acme Corp` (7+3 case-variants) vs `ACME Corporation` (4).**
   Genuinely unclear if the same company. The tool folds only the case/whitespace
   variants into `Acme Corp` and **leaves `ACME Corporation` untouched** — it does not
   silently merge two plausibly-distinct entities. Human judgment call. ✅ behaves correctly.

2. **Ambiguous country `Slovia`.** Reconciles to `Slovakia` (0.857) and `Slovenia`
   essentially tied → **margin 0.0 < 0.03 ⇒ ABSTAIN**. Surfaces in the
   `uncertain_canonicalization` review flag. The hero wow-moment.

3. **Ambiguous country `Austrai`.** `Australia` (0.875) vs `Austria` → **margin 0.018 < 0.03
   ⇒ ABSTAIN**. Second genuine geographic ambiguity.

Bonus near-miss abstentions also fire (`Indai`→India 0.80 < 0.84 threshold; `Frnace`,
`Brasil`) — they look like typos but don't clear the confidence bar, so they're flagged,
not guessed.

## Contrast: clean auto-fixes vs abstentions (the trust story)
- **Confident typos auto-fix:** `Nigeia`→Nigeria, `germny`→Germany, `Calfornia`→California
  (high score, wide margin to the next candidate).
- **Ambiguous/low-confidence ABSTAIN:** `Slovia`, `Austrai`, `Indai` — left unchanged,
  raised as `country | uncertain_canonicalization` for Maria to decide.
That split (fix the obvious, flag the genuine judgment calls) is the demo's whole pitch.
