# Paired Bench — shipped system on every cell-aligned pair

Churn-neutral repairs metric + variant-class recall; `seen` = source fed
the champion's training mix (flagged, not hidden).

| dataset | seen | rows×cols | errors | variant | F1 | precision | recall | VR | damage |
|---|---|---|---|---|---|---|---|---|---|
| cleanml_movie | ✓ | 9329×8 | 4779 | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0001 |
| dblp_scholar |  | 2408×4 | 3099 | 3099 | 0.0 | 0.001 | 0.0 | 0.0 | 0.2307 |
| dgov_2_10_budget_presentation_award_summary |  | 16×6 | 9 | 9 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| fodors_zagats | ✓ | 112×6 | 206 | 206 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0536 |
| rayyan |  | 1000×11 | 948 | 171 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1445 |
| zeroed_tax100k |  | 20000×15 | 952 | 117 | 0.0 | 0.0 | 0.006 | 0.051 | 0.0822 |
| dblp_acm |  | 2224×4 | 2128 | 2128 | 0.001 | 0.333 | 0.0 | 0.0 | 0.0001 |
| ed2_restaurants |  | 20000×15 | 309 | 76 | 0.001 | 0.0 | 0.026 | 0.105 | 0.0716 |
| beers | ✓ | 2410×11 | 4362 | 693 | 0.026 | 0.042 | 0.019 | 0.117 | 0.0041 |
| flights |  | 2376×7 | 4920 | 1049 | 0.044 | 0.078 | 0.03 | 0.142 | 0.082 |
| hospital | ✓ | 1000×20 | 509 | 379 | 0.092 | 0.056 | 0.257 | 0.346 | 0.1078 |
| zeroed_billionaire |  | 2614×22 | 5248 | 1146 | 0.106 | 0.264 | 0.066 | 0.303 | 0.0007 |
| dgov_ah_provisional_diabetes_death_counts_for |  | 226×16 | 142 | 141 | 0.27 | 0.202 | 0.408 | 0.411 | 0.0512 |
| cleanml_company | ✓ | 20000×9 | 65 | 65 | 0.272 | 0.168 | 0.708 | 0.708 | 0.0013 |
| gidcl_imdb |  | 20000×6 | 13320 | 7890 | 0.436 | 0.488 | 0.394 | 0.666 | 0.0296 |
| dgov_3_09_census_acs_post_secondary_education |  | 53×17 | 82 | 82 | 0.5 | 0.933 | 0.341 | 0.341 | 0.0 |
| dgov_access_control |  | 4928×13 | 4180 | 4161 | 0.543 | 0.932 | 0.383 | 0.385 | 0.0 |
| dgov_305b_assessed_lake_2020 |  | 182×23 | 442 | 424 | 0.556 | 0.766 | 0.437 | 0.455 | 0.0139 |
| movies_1 | ✓ | 7390×17 | 7006 | 5567 | 0.707 | 0.643 | 0.786 | 0.989 | 0.0222 |
