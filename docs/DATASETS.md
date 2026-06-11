# Dataset inventory — every source the system trains on, evaluates on, or must clean

Stage-3 consolidated registry (2026-06-11). Assignment discipline: a source is
TRAIN, EVAL, or BENCH — never both sides of train/eval.

## Paired dirty/clean (27 — eval/paired_bench.py → docs/PAIRED_BENCH.md)

| source | origin | license | assignment | notes |
|---|---|---|---|---|
| hospital, beers, movies_1 | Raha (BigDaMa) | Apache-2.0 | TRAIN | champion mix since v6 |
| flights, rayyan | Raha | Apache-2.0 | EVAL (GEN) | held-out real errors |
| tax | Raha | Apache-2.0 | unused | numeric-heavy, huge |
| ed2_restaurants | BigDaMa ED2 | research | EVAL (GEN) | real NYC variants; errors past row 2k |
| fodors_zagats | Magellan EM | BSD-ish data | TRAIN | variant-masked EM table |
| dblp_acm, dblp_scholar | Magellan EM | research | BENCH only | out-of-regime (unique titles / convention-mismatch gold) |
| cleanml_company, cleanml_movie | CleanML | research | TRAIN | Company = org canon |
| gidcl_imdb | SICS-FRC GIDCL | none stated | TRAIN (v9+) | 1M-row pair; 57k errors; subset 86k rows |
| zeroed_billionaire, zeroed_tax100k | WelkinNi/ZeroED | none stated | BENCH | injected; rich categoricals |
| dgov_* (5 tables) | LUH-DBS Matelda | Apache-2.0 | BENCH | real data.gov tables, injected typos (6,692 more available) |
| tt_* (8 tables) | ToughTables 2T_WD | CC-BY-4.0 | BENCH | gold-anchored entity misspellings, 370–33.5k corrections each |

## Wild messy tables (35 — eval/wild_bench.py → docs/WILD_BENCH.md)

24 portal tables (training/unpaired_sources.json cache: NYC/Chicago/SF/LA/Seattle/TX/WA
portals, spotify, billboard, titanic, worldcities, airlines) + 12 stage-3 additions
(training/harvest_wild.py): bx_books (mojibake), salary_survey, fec_indiv80 (PII,
headerless), acnc_charities (AU), uk_price_paid (headerless UK), irs_eo1,
glassdoor_jobs (multiline cells), paris_trees (FR), online_retail, bl_flickr_books,
open_food_facts (211 cols), ct_real_estate. Backlog: CMS doctors (API 400), NHTSA
FLAT_CMPL (multi-GB), Canada contracts (627MB).

## Alias vocabularies (training generator material)

| vocab | size | license | regime |
|---|---|---|---|
| toughtables_aliases | 49,629 | CC-BY-4.0 | real entity misspellings (gold-anchored) |
| musicbrainz_hint_aliases | 34,017 | CC0 | community-recorded artist misspellings |
| rxnorm_aliases | 17,701 | public domain | drug name synonyms |
| ror_aliases | 73k orgs | CC0 | research orgs |
| geonames_city_aliases | 80k cities | CC-BY | city aliases |
| wikidata_company_aliases | 10.2k | CC0 | company aliases |
| onet_jobtitle_aliases | 1,016 | CC-BY-4.0 | job titles |
| nickname_aliases | 555 | Apache-2.0 | first names |
| openflights_airports | 7,698 | ODbL/DbCL | airports reference |
| libpostal_aliases | — | MIT | address abbreviations |

## Measured conclusions that govern future widening

1. Pre-paired corpus discovery is SATURATED (3 verified hunts) — synthesis from
   vocabularies is the widening path.
2. Pair volume / vocab training does NOT move held-out generalization (v7–v9, 4
   retrains + tt-transfer test): the planner's value_counts cap (80) structurally
   hides high-cardinality dirty cells. The unlock is architectural: error-suspect /
   windowed profiling and cross-row entity voting.
3. The deterministic side (grounding + ops + verifier union) carries never-seen
   tables today; every op added from a measured regime (normalize_punctuation)
   moved GEN; convention/encoding ops are the cheapest remaining wins.
