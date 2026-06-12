# GitTables N=250 audit — trust contract at scale

Shipped pipeline over 239 real GitHub tables (Matelda GitTables-subsets,
Apache-2.0). IMPORTANT framing: this subset is a CLEAN LAKE (dirty == clean for
238/239 tables), so the repair-F1 dimension is void and `macro_damage` is NOT
damage — it is an INTERVENTION-RATE upper bound (any semantic normalization the
pipeline performs counts against gold=input, including intended format parsing).
What this audit certifies: robustness (0 pipeline failures), schema validity
(239/239), and ZERO silent edits across 239 arbitrary real-world tables — the
trust contract at scale. The ~5.5% intervention rate (43 tables untouched) is
the conservative measure of how much the pipeline chooses to act on arbitrary
tables.

| metric | value |
|---|---|
| tables_audited | 239 |
| pipeline_failures | 0 |
| plan_valid | 239 |
| tables_with_silent_edits | 0 |
| tables_with_errors | 1 |
| macro_f1_on_errored | 0.0 |
| macro_damage | 0.055 |
| zero_damage_tables | 43 |
| seconds | 796.9 |
