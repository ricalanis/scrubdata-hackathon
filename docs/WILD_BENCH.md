# Wild Bench — can the shipped system clean real-world tables?

Behavioral audit + seeded inject-recovery per dataset (eval/wild_bench.py).

| dataset | domain | rows×cols | valid | changes | flags | PII | silent | typo | ocr | case | ws | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| airlines | aviation | 56×8 | ✓ | 413 | 1 | 1 | 0 | — | — | — | — | — |
| billboard | music-billboard | 317×83 | ✓ | 36222 | 3 | 2 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| acnc_charities | nonprofits-au | 800×69 | ✓ | 43268 | 4 | 1 | 0 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 |
| open_food_facts | food-products | 800×211 | ✓ | 27115 | 34 | 5 | 0 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 |
| biz_sf | sf-business | 800×38 | ✓ | 8060 | 12 | 1 | 0 | 0.02 | 0.05 | 0.02 | 0.07 | 0.04 |
| irs_eo1 | nonprofits-us | 800×28 | ✓ | 16953 | 5 | 3 | 0 | 0.04 | 0.03 | 0.03 | 0.15 | 0.06 |
| permits_nyc | construction | 800×60 | ✓ | 16762 | 25 | 3 | 0 | 0.03 | 0.04 | 0.04 | 0.13 | 0.06 |
| pawnbrokers_nyc | business | 800×31 | ✓ | 8494 | 8 | 2 | 0 | 0.06 | 0.08 | 0.05 | 0.11 | 0.08 |
| proptax_sf | real-estate | 800×46 | ✓ | 9302 | 3 | 3 | 0 | 0.06 | 0.06 | 0.07 | 0.12 | 0.08 |
| biz_chicago | business-licenses | 800×37 | ✓ | 12808 | 9 | 2 | 0 | 0.05 | 0.06 | 0.06 | 0.15 | 0.08 |
| permits_seattle | seattle-permits | 800×40 | ✓ | 6878 | 9 | 2 | 0 | 0.08 | 0.13 | 0.09 | 0.14 | 0.11 |
| restaurants_nyc | restaurants | 800×27 | ✓ | 7742 | 6 | 4 | 0 | 0.07 | 0.08 | 0.09 | 0.20 | 0.11 |
| titanic | passengers | 800×12 | ✓ | 5722 | 1 | 0 | 0 | 0.00 | 0.00 | 0.09 | 0.40 | 0.12 |
| biz_la | la-business | 800×16 | ✓ | 2726 | 9 | 3 | 0 | 0.15 | 0.09 | 0.10 | 0.21 | 0.14 |
| schools_nyc | education | 800×41 | ✓ | 14387 | 7 | 5 | 0 | 0.08 | 0.14 | 0.12 | 0.22 | 0.14 |
| online_retail | ecommerce-uk | 800×8 | ✓ | 3413 | 1 | 0 | 0 | 0.26 | 0.01 | 0.01 | 0.30 | 0.14 |
| film_nyc | film | 800×14 | ✓ | 3049 | 3 | 0 | 0 | 0.14 | 0.16 | 0.11 | 0.23 | 0.16 |
| salary_survey | survey | 800×18 | ✓ | 4142 | 5 | 0 | 0 | 0.12 | 0.20 | 0.13 | 0.26 | 0.18 |
| restaurants_sf | sf-restaurants | 800×22 | ✓ | 6002 | 6 | 2 | 0 | 0.15 | 0.15 | 0.16 | 0.26 | 0.18 |
| alcohol_tx | alcohol-bars | 800×24 | ✓ | 8518 | 9 | 1 | 0 | 0.14 | 0.09 | 0.17 | 0.38 | 0.20 |
| contractors_chi | contractors | 800×116 | ✓ | 20213 | 22 | 2 | 0 | 0.17 | 0.20 | 0.16 | 0.33 | 0.21 |
| fhv_nyc | transport | 800×23 | ✓ | 3789 | 4 | 2 | 0 | 0.10 | 0.30 | 0.14 | 0.36 | 0.23 |
| uk_price_paid | real-estate-uk | 800×16 | ✓ | 1662 | 8 | 0 | 0 | 0.14 | 0.17 | 0.26 | 0.42 | 0.25 |
| food_chicago | food-inspections | 800×17 | ✓ | 2790 | 6 | 0 | 0 | 0.17 | 0.25 | 0.23 | 0.38 | 0.26 |
| bx_books | books | 800×8 | ✓ | 1650 | 3 | 1 | 0 | 0.22 | 0.22 | 0.16 | 0.51 | 0.28 |
| bl_flickr_books | library | 800×15 | ✓ | 1769 | 6 | 1 | 0 | 0.19 | 0.28 | 0.22 | 0.43 | 0.28 |
| svc311_nyc | complaints | 800×44 | ✓ | 6299 | 16 | 2 | 0 | 0.23 | 0.30 | 0.23 | 0.37 | 0.28 |
| spotify | music | 800×23 | ✓ | 4669 | 3 | 1 | 0 | 0.20 | 0.28 | 0.30 | 0.36 | 0.28 |
| glassdoor_jobs | job-listings | 800×14 | ✓ | 1713 | 6 | 0 | 0 | 0.20 | 0.29 | 0.22 | 0.43 | 0.29 |
| ct_real_estate | real-estate-us | 800×14 | ✓ | 4840 | 4 | 0 | 0 | 0.23 | 0.29 | 0.24 | 0.40 | 0.29 |
| worldcities | geography | 800×4 | ✓ | 914 | 2 | 0 | 0 | 0.41 | 0.11 | 0.22 | 0.69 | 0.36 |
| fec_indiv80 | political-finance | 800×21 | ✓ | 4375 | 4 | 2 | 0 | 0.20 | 0.24 | 0.35 | 0.87 | 0.41 |
| payroll_nyc | jobs | 800×17 | ✓ | 4587 | 3 | 2 | 0 | 0.45 | 0.56 | 0.42 | 0.73 | 0.54 |
| paris_trees | urban-fr | 800×16 | ✓ | 3305 | 5 | 1 | 0 | 0.43 | 0.54 | 0.55 | 0.73 | 0.56 |
| ev_wa | vehicles | 800×16 | ✓ | 4085 | 5 | 2 | 0 | 0.50 | 0.56 | 0.48 | 0.91 | 0.61 |
