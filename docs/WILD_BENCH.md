# Wild Bench — can the shipped system clean real-world tables?

Behavioral audit + seeded inject-recovery per dataset (eval/wild_bench.py).

| dataset | domain | rows×cols | valid | changes | flags | PII | silent | typo | ocr | case | ws | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| airlines | aviation | 56×8 | ✓ | 392 | 0 | 1 | 0 | — | — | — | — | — |
| billboard | music-billboard | 317×83 | ✓ | 36222 | 1 | 2 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| acnc_charities | nonprofits-au | 800×69 | ✓ | 44865 | 1 | 1 | 0 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |
| open_food_facts | food-products | 800×211 | ✓ | 25768 | 5 | 5 | 0 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 |
| biz_sf | sf-business | 800×38 | ✓ | 9660 | 3 | 1 | 0 | 0.02 | 0.04 | 0.02 | 0.06 | 0.04 |
| permits_nyc | construction | 800×60 | ✓ | 21293 | 10 | 3 | 0 | 0.02 | 0.03 | 0.03 | 0.09 | 0.04 |
| proptax_sf | real-estate | 800×46 | ✓ | 9302 | 0 | 3 | 0 | 0.03 | 0.04 | 0.07 | 0.12 | 0.07 |
| biz_la | la-business | 800×16 | ✓ | 4316 | 2 | 3 | 0 | 0.08 | 0.05 | 0.05 | 0.12 | 0.07 |
| pawnbrokers_nyc | business | 800×31 | ✓ | 9266 | 2 | 2 | 0 | 0.06 | 0.07 | 0.05 | 0.12 | 0.07 |
| biz_chicago | business-licenses | 800×37 | ✓ | 13601 | 3 | 2 | 0 | 0.04 | 0.05 | 0.06 | 0.15 | 0.08 |
| contractors_chi | contractors | 800×116 | ✓ | 24996 | 6 | 2 | 0 | 0.06 | 0.08 | 0.06 | 0.13 | 0.08 |
| restaurants_nyc | restaurants | 800×27 | ✓ | 8520 | 1 | 4 | 0 | 0.07 | 0.08 | 0.09 | 0.20 | 0.11 |
| permits_seattle | seattle-permits | 800×40 | ✓ | 7667 | 0 | 2 | 0 | 0.08 | 0.13 | 0.09 | 0.14 | 0.11 |
| titanic | passengers | 800×12 | ✓ | 5722 | 0 | 0 | 0 | 0.00 | 0.00 | 0.09 | 0.40 | 0.12 |
| film_nyc | film | 800×14 | ✓ | 3849 | 0 | 0 | 0 | 0.11 | 0.13 | 0.08 | 0.18 | 0.13 |
| schools_nyc | education | 800×41 | ✓ | 14385 | 1 | 5 | 0 | 0.09 | 0.14 | 0.13 | 0.22 | 0.15 |
| irs_eo1 | nonprofits-us | 800×28 | ✓ | 16026 | 1 | 3 | 0 | 0.10 | 0.07 | 0.08 | 0.35 | 0.15 |
| online_retail | ecommerce-uk | 800×8 | ✓ | 3405 | 0 | 0 | 0 | 0.27 | 0.01 | 0.01 | 0.31 | 0.15 |
| bx_books | books | 800×8 | ✓ | 1609 | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 | 0.63 | 0.16 |
| alcohol_tx | alcohol-bars | 800×24 | ✓ | 10115 | 1 | 1 | 0 | 0.08 | 0.06 | 0.12 | 0.38 | 0.16 |
| salary_survey | survey | 800×18 | ✓ | 4128 | 2 | 0 | 0 | 0.12 | 0.20 | 0.14 | 0.26 | 0.18 |
| restaurants_sf | sf-restaurants | 800×22 | ✓ | 6793 | 0 | 2 | 0 | 0.15 | 0.15 | 0.18 | 0.25 | 0.18 |
| svc311_nyc | complaints | 800×44 | ✓ | 7161 | 1 | 2 | 0 | 0.17 | 0.23 | 0.18 | 0.27 | 0.21 |
| fhv_nyc | transport | 800×23 | ✓ | 3774 | 0 | 2 | 0 | 0.11 | 0.30 | 0.14 | 0.36 | 0.23 |
| food_chicago | food-inspections | 800×17 | ✓ | 3584 | 0 | 0 | 0 | 0.17 | 0.20 | 0.23 | 0.38 | 0.24 |
| uk_price_paid | real-estate-uk | 800×16 | ✓ | 1659 | 3 | 0 | 0 | 0.14 | 0.17 | 0.26 | 0.43 | 0.25 |
| glassdoor_jobs | job-listings | 800×14 | ✓ | 1691 | 2 | 0 | 0 | 0.18 | 0.29 | 0.22 | 0.44 | 0.28 |
| spotify | music | 800×23 | ✓ | 4664 | 0 | 1 | 0 | 0.20 | 0.28 | 0.30 | 0.36 | 0.29 |
| ct_real_estate | real-estate-us | 800×14 | ✓ | 3875 | 1 | 0 | 0 | 0.23 | 0.29 | 0.24 | 0.40 | 0.29 |
| bl_flickr_books | library | 800×15 | ✓ | 1741 | 1 | 1 | 0 | 0.20 | 0.30 | 0.24 | 0.45 | 0.30 |
| worldcities | geography | 800×4 | ✓ | 914 | 2 | 0 | 0 | 0.41 | 0.11 | 0.22 | 0.69 | 0.36 |
| fec_indiv80 | political-finance | 800×21 | ✓ | 4841 | 1 | 2 | 0 | 0.29 | 0.22 | 0.46 | 0.88 | 0.46 |
| payroll_nyc | jobs | 800×17 | ✓ | 4587 | 2 | 2 | 0 | 0.45 | 0.56 | 0.41 | 0.73 | 0.53 |
| paris_trees | urban-fr | 800×16 | ✓ | 3291 | 3 | 1 | 0 | 0.43 | 0.54 | 0.56 | 0.73 | 0.57 |
| ev_wa | vehicles | 800×16 | ✓ | 4885 | 3 | 2 | 0 | 0.59 | 0.67 | 0.70 | 0.90 | 0.72 |
