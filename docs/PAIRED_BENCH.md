# Paired Bench — shipped system on every cell-aligned pair

Churn-neutral repairs metric + variant-class recall; `seen` = source fed
the champion's training mix (flagged, not hidden).

| dataset | seen | rows×cols | errors | variant | F1 | precision | recall | VR | damage |
|---|---|---|---|---|---|---|---|---|---|
| cleanml_movie | ✓ | 9329×8 | 4779 | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0003 |
| dgov_2_10_budget_presentation_award_summary |  | 16×6 | 9 | 9 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| dgov_emergency_operating_center_tools |  | 7×3 | 4 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5882 |
| dgov_illinois_obesity_by_county |  | 102×5 | 17 | 17 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| fodors_zagats | ✓ | 112×6 | 206 | 206 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0536 |
| rayyan |  | 1000×11 | 948 | 171 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1463 |
| zeroed_tax100k |  | 20000×15 | 952 | 117 | 0.0 | 0.0 | 0.006 | 0.051 | 0.0822 |
| ed2_restaurants |  | 20000×15 | 309 | 76 | 0.001 | 0.0 | 0.026 | 0.105 | 0.0719 |
| dblp_scholar |  | 2408×4 | 3099 | 3099 | 0.002 | 0.003 | 0.001 | 0.001 | 0.2333 |
| dblp_acm |  | 2224×4 | 2128 | 2128 | 0.003 | 0.103 | 0.001 | 0.001 | 0.0033 |
| tt_cn5wvwhh |  | 8302×5 | 370 | 370 | 0.021 | 0.046 | 0.014 | 0.014 | 0.0025 |
| beers | ✓ | 2410×11 | 4362 | 693 | 0.026 | 0.042 | 0.019 | 0.117 | 0.0047 |
| dgov_mva_vehicle_sales_counts_by_month_for_ca |  | 248×6 | 43 | 24 | 0.042 | 0.2 | 0.023 | 0.042 | 0.0 |
| flights |  | 2376×7 | 4920 | 1049 | 0.044 | 0.078 | 0.03 | 0.142 | 0.082 |
| dgov_median_household_income |  | 174×19 | 138 | 83 | 0.084 | 0.057 | 0.159 | 0.265 | 0.1098 |
| hospital | ✓ | 1000×20 | 509 | 379 | 0.092 | 0.056 | 0.257 | 0.346 | 0.1078 |
| zeroed_billionaire |  | 2614×22 | 5248 | 1146 | 0.103 | 0.232 | 0.067 | 0.305 | 0.0042 |
| dgov_field_listings |  | 122×20 | 317 | 250 | 0.109 | 0.136 | 0.091 | 0.116 | 0.0532 |
| dgov_allegheny_county_tobacco_vendors |  | 1248×12 | 2392 | 2109 | 0.173 | 0.139 | 0.23 | 0.261 | 0.2115 |
| dgov_grocery_stores_2013 |  | 506×17 | 420 | 332 | 0.174 | 0.464 | 0.107 | 0.136 | 0.0009 |
| dgov_jefferson_county_ky_post_offices |  | 32×9 | 26 | 26 | 0.194 | 0.13 | 0.385 | 0.385 | 0.2366 |
| dgov_louisville_metro_ky_inspection_results_p |  | 521×18 | 1126 | 1044 | 0.224 | 0.281 | 0.187 | 0.201 | 0.0635 |
| cleanml_company | ✓ | 20000×9 | 65 | 65 | 0.243 | 0.147 | 0.708 | 0.708 | 0.0015 |
| dgov_ah_provisional_diabetes_death_counts_for |  | 226×16 | 142 | 141 | 0.27 | 0.202 | 0.408 | 0.411 | 0.0512 |
| dgov_louisville_metro_ky_permitted_hotels_and |  | 131×13 | 191 | 182 | 0.277 | 0.276 | 0.277 | 0.291 | 0.0886 |
| dgov_la_county_covid_cases |  | 975×14 | 579 | 579 | 0.34 | 0.983 | 0.206 | 0.206 | 0.0 |
| tt_co23z7go |  | 15477×4 | 33542 | 33542 | 0.36 | 0.929 | 0.223 | 0.223 | 0.0004 |
| dgov_legislative_bridge_names |  | 252×16 | 415 | 396 | 0.363 | 0.615 | 0.258 | 0.27 | 0.0091 |
| dgov_health_conditions_among_children_under_a |  | 2744×16 | 2900 | 2844 | 0.422 | 0.352 | 0.528 | 0.539 | 0.0569 |
| gidcl_imdb | ✓ | 20000×6 | 13320 | 7890 | 0.438 | 0.489 | 0.396 | 0.669 | 0.0297 |
| tt_uma1dnf6 |  | 8302×5 | 5080 | 5080 | 0.442 | 0.911 | 0.292 | 0.292 | 0.0026 |
| dgov_medicare_part_d_opioid_prescribing_rates |  | 677×17 | 547 | 547 | 0.447 | 0.775 | 0.314 | 0.314 | 0.0026 |
| dgov_access_control |  | 4928×13 | 4180 | 4161 | 0.551 | 0.933 | 0.391 | 0.392 | 0.0 |
| dgov_3_09_census_acs_post_secondary_education |  | 53×17 | 82 | 82 | 0.552 | 0.941 | 0.39 | 0.39 | 0.0 |
| dgov_305b_assessed_lake_2020 |  | 182×23 | 442 | 424 | 0.556 | 0.766 | 0.437 | 0.455 | 0.0139 |
| dgov_national_obesity_by_state_1 |  | 52×5 | 13 | 13 | 0.7 | 1.0 | 0.538 | 0.538 | 0.0 |
| movies_1 | ✓ | 7390×17 | 7006 | 5567 | 0.704 | 0.637 | 0.786 | 0.989 | 0.0228 |
| tt_3n6s2fcx |  | 9396×3 | 9510 | 9510 | 0.955 | 0.998 | 0.916 | 0.916 | 0.0 |
| tt_2zwsmotj |  | 10855×3 | 10977 | 10977 | 0.956 | 0.997 | 0.918 | 0.918 | 0.0 |
| tt_8yinkydr |  | 14008×3 | 14188 | 14188 | 0.956 | 0.997 | 0.918 | 0.918 | 0.0 |
| tt_dvnkv0xu |  | 15477×4 | 15676 | 15676 | 0.956 | 0.997 | 0.919 | 0.919 | 0.0 |
| tt_00e2h310 |  | 12285×3 | 12433 | 12433 | 0.957 | 0.998 | 0.919 | 0.919 | 0.0 |
