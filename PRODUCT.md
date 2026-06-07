# ScrubData — Product Research & Spec

> What does an office worker actually mean by "just clean my data"? This doc
> pins down the expectations so the cleaning-plan schema and UX aren't guesses.
> (Living doc — refine when the deep-research workflows land.)

## 1. The user & the moment

**Who:** an operations / sales-ops / finance / admin person. Lives in
spreadsheets exported from a CRM, an ERP, a Google Form, a POS, a bank portal.
Not a pandas user. Competent with Excel but doesn't want to write `=PROPER()`
across 40 columns or learn Power Query.

**The moment of pain:** they exported a file to do their actual job —
build a report, upload to another system, send a mail-merge, reconcile numbers —
and the file is dirty enough that the next step breaks or lies. The import fails,
the pivot double-counts, the vlookup misses, the "total revenue" is wrong because
amounts are text.

**What they want:** drop the file in, get a *trustworthy* clean file back, and
a plain sentence telling them what was wrong so they can vouch for it to their
boss. They do **not** want 30 config toggles. Hands-off is the whole pitch.

**What they fear (must design against):** that the tool silently changed
something it shouldn't have. Trust is the product. Every change must be
**visible, explained, and reversible**.

## 2. Taxonomy of "dirty" — what we must detect & fix

Grouped by how an office worker would describe it. This list *is* the operation
set the planner emits and the executor implements.

### A. Structural / table-level
- **Exact duplicate rows** — "this person is in here 3 times."
- **Near-duplicate rows** — same entity, trivial differences (later/stretch).
- **Empty rows & empty columns** — junk from the export.
- **Header problems** — header not in row 1, merged cells, `Unnamed: 0`,
  duplicated column names, units baked into headers (`Amount (USD)`).
- **Inconsistent column naming** — `First Name` vs `first_name` (normalize to
  snake_case as an option, off by default — it's a rename, higher-trust-risk).

### B. Whitespace & casing (the silent killers behind failed joins)
- Leading/trailing whitespace; doubled internal spaces; non-breaking spaces.
- Inconsistent casing (`ACME`, `Acme`, `acme corp`).
- Invisible characters (zero-width, BOM), smart quotes.

### C. Missing values, disguised
- Real blanks **plus** disguised nulls: `N/A`, `na`, `-`, `--`, `null`, `None`,
  `#N/A`, `TBD`, `?`, `0` (context-dependent — risky, don't auto-assume).
- Decision: normalize disguised nulls → true missing; **imputation is opt-in**,
  never silent (filling values is a claim about reality).

### D. Type & format inconsistency (where the model earns its keep)
- **Numbers stored as text:** `"$1,200.50"`, `"1.200,50"` (EU), `"(500)"`
  (accounting negative), `"12%"`, `"1,2k"`.
- **Dates in mixed formats:** `2023-01-05`, `01/05/2023`, `5 Jan 2023`,
  `Jan-23`, Excel serial `44931`. Ambiguous DMY vs MDY must be detected, not
  guessed blindly — infer from the column's evidence, flag if undecidable.
- **Booleans:** `Yes/No`, `Y/N`, `TRUE/FALSE`, `1/0`, `T/F`, `✓`.
- **Phone numbers:** wildly inconsistent; standardize to E.164-ish where region
  is inferable, else just strip to digits + canonical format.
- **Emails:** casing, whitespace, obvious typos (`@gmial.com`), trailing junk.

### E. Categorical canonicalization (the headline AI feature)
- Inconsistent labels for the same thing: `USA / U.S.A. / United States / us`,
  `M/F vs Male/Female`, `NY / New York / new york`, status fields, product
  names. Rules can't enumerate these — **the small model proposes the mapping**,
  the executor applies it, the report shows the mapping for approval.

### F. Validity / anomaly flags (flag, don't auto-delete)
- Out-of-range numbers (age 999, negative price), impossible dates (1899-12-31
  Excel epoch), malformed emails/phones, values that don't match the column's
  inferred type. Default action = **flag in the report**, not silent edit.

## 3. The trust contract (design principles)

1. **Visible** — every operation appears in a before/after diff and the report.
2. **Explained** — plain-English rationale per operation ("standardized 4 date
   formats into ISO `YYYY-MM-DD`").
3. **Conservative by default** — destructive/assumptive ops (imputation, row
   deletion beyond exact dups, renames) are surfaced as suggestions, applied
   only if the user keeps them on. Safe ops (trim whitespace, normalize disguised
   nulls, parse types) are on by default.
4. **Reversible** — original file untouched; output is a new file + a machine-
   readable plan the user could replay or undo.
5. **No config to start** — sensible defaults run immediately on upload; the
   plan is editable *after* the user sees it, not a wall of options before.

## 4. Competitive landscape (what to learn / what to beat)

| Tool | What it does well | Why an office worker bounces |
|------|-------------------|------------------------------|
| **Excel / Power Query** | Ubiquitous, trusted | Manual; canonicalization is hand-built; steep |
| **OpenRefine** | Powerful clustering/canonicalization (key-collision, kNN) | Intimidating UI, GREL expressions, local Java app |
| **ydata-profiling / pandas-profiling** | Great *profiling* report | Diagnoses, doesn't *fix* |
| **Trifacta / Tableau Prep / Alteryx** | Visual prep pipelines | Enterprise, paid, config-heavy |
| **OpenRefine reconciliation** | Entity canonicalization | Manual, needs setup |

**Our wedge:** OpenRefine's clustering *automated and explained by a small
model*, with zero config and a one-screen trust-preserving UX. We borrow
OpenRefine's clustering idea but the model proposes the clusters/mappings and
narrates them, so the user never learns a tool — they just approve sentences.

## 5. Cleaning-plan schema (v0 — drives the mock & later the model)

The model outputs this JSON; the executor consumes it. Designed so the model
only does *semantic/fuzzy* judgment, and all execution is deterministic.

```json
{
  "dataset_summary": "Contacts export, 38 rows × 9 cols; sales-lead data.",
  "table_operations": [
    {"op": "drop_exact_duplicates", "rationale": "5 identical rows."},
    {"op": "drop_empty_rows"},
    {"op": "drop_empty_columns", "columns": ["notes2"]}
  ],
  "columns": [
    {
      "name": "country",
      "detected_semantic_type": "country",
      "issues": ["inconsistent_categories", "whitespace", "casing"],
      "operations": [
        {"op": "strip_whitespace"},
        {"op": "canonicalize_categories",
         "mapping": {"usa": "United States", "u.s.a.": "United States",
                     "us": "United States", "uk": "United Kingdom"},
         "rationale": "Unified 4 spellings into 2 canonical country names."}
      ],
      "confidence": 0.93
    },
    {
      "name": "amount",
      "detected_semantic_type": "currency",
      "issues": ["numeric_stored_as_text", "currency_symbols"],
      "operations": [
        {"op": "parse_currency", "rationale": "Stripped $ and thousands separators; → float."}
      ],
      "confidence": 0.97
    }
  ],
  "flags": [
    {"column": "age", "row_hint": "value 999", "issue": "out_of_range",
     "action": "flag_only", "rationale": "Likely placeholder; left for human review."}
  ]
}
```

### Operation vocabulary (executor must implement)
Safe-by-default: `strip_whitespace`, `collapse_internal_whitespace`,
`normalize_disguised_nulls`, `standardize_case`, `parse_currency`,
`parse_number`, `parse_percent`, `parse_date`, `standardize_boolean`,
`standardize_phone`, `normalize_email`, `drop_exact_duplicates`,
`drop_empty_rows`, `drop_empty_columns`, `canonicalize_categories`.
Opt-in (assumptive): `impute_missing`, `drop_near_duplicates`,
`rename_columns_snake_case`, `coerce_outliers`.
Flag-only: `flag_out_of_range`, `flag_invalid_format`, `flag_type_mismatch`.

## 6. Success metric for the demo (Backyard AI judging)

A real office person uploads a real ugly export, clicks one button, and says
"oh thank god" — then trusts the result enough to use it, because the report
told them exactly what changed. That sentence is the bar.
