# Wild Bench — can the shipped system clean real-world tables?

Behavioral audit + seeded inject-recovery per dataset (eval/wild_bench.py).

| dataset | domain | rows×cols | valid | changes | flags | PII | silent | typo | ocr | case | ws | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| airlines | aviation | 56×8 | ✓ | 413 | 1 | 1 | 0 | — | — | — | — | — |
| billboard | music-billboard | 317×83 | ✓ | 36222 | 3 | 2 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| acnc_charities | nonprofits-au | 800×69 | ✓ | 44869 | 4 | 1 | 0 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |
| open_food_facts | food-products | 800×211 | ✓ | 27210 | 35 | 5 | 0 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 |
| biz_sf | sf-business | 800×38 | ✓ | 9660 | 13 | 1 | 0 | 0.02 | 0.05 | 0.02 | 0.06 | 0.04 |
| permits_nyc | construction | 800×60 | ✓ | 21580 | 20 | 3 | 0 | 0.02 | 0.03 | 0.02 | 0.08 | 0.04 |
| proptax_sf | real-estate | 800×46 | ✓ | 9302 | 3 | 3 | 0 | 0.03 | 0.04 | 0.06 | 0.12 | 0.07 |
| biz_la | la-business | 800×16 | ✓ | 4331 | 9 | 3 | 0 | 0.08 | 0.05 | 0.05 | 0.12 | 0.07 |
| pawnbrokers_nyc | business | 800×31 | ✓ | 9300 | 8 | 2 | 0 | 0.06 | 0.07 | 0.05 | 0.11 | 0.07 |
| biz_chicago | business-licenses | 800×37 | ✓ | 13611 | 8 | 2 | 0 | 0.05 | 0.06 | 0.06 | 0.15 | 0.08 |
| contractors_chi | contractors | 800×116 | ✓ | 25038 | 22 | 2 | 0 | 0.06 | 0.07 | 0.06 | 0.13 | 0.08 |
| restaurants_nyc | restaurants | 800×27 | ✓ | 8545 | 5 | 4 | 0 | 0.07 | 0.08 | 0.09 | 0.20 | 0.11 |
| permits_seattle | seattle-permits | 800×40 | ✓ | 7681 | 9 | 2 | 0 | 0.08 | 0.13 | 0.09 | 0.14 | 0.11 |
| titanic | passengers | 800×12 | ✓ | 5722 | 1 | 0 | 0 | 0.00 | 0.00 | 0.09 | 0.40 | 0.12 |
| film_nyc | film | 800×14 | ✓ | 3849 | 2 | 0 | 0 | 0.11 | 0.13 | 0.08 | 0.18 | 0.13 |
| online_retail | ecommerce-uk | 800×8 | ✓ | 3421 | 1 | 0 | 0 | 0.26 | 0.01 | 0.01 | 0.30 | 0.14 |
| irs_eo1 | nonprofits-us | 800×28 | ✓ | 16031 | 6 | 3 | 0 | 0.10 | 0.07 | 0.08 | 0.34 | 0.14 |
| schools_nyc | education | 800×41 | ✓ | 14393 | 7 | 5 | 0 | 0.09 | 0.14 | 0.13 | 0.22 | 0.15 |
| salary_survey | survey | 800×18 | ✓ | 4155 | 4 | 0 | 0 | 0.12 | 0.19 | 0.13 | 0.26 | 0.17 |
| restaurants_sf | sf-restaurants | 800×22 | ✓ | 6807 | 5 | 2 | 0 | 0.15 | 0.15 | 0.18 | 0.25 | 0.18 |
| alcohol_tx | alcohol-bars | 800×24 | ✓ | 10120 | 7 | 1 | 0 | 0.14 | 0.09 | 0.17 | 0.37 | 0.20 |
| svc311_nyc | complaints | 800×44 | ✓ | 7915 | 14 | 2 | 0 | 0.17 | 0.23 | 0.17 | 0.26 | 0.21 |
| fhv_nyc | transport | 800×23 | ✓ | 3790 | 4 | 2 | 0 | 0.10 | 0.30 | 0.14 | 0.36 | 0.23 |
| food_chicago | food-inspections | 800×17 | ✓ | 3599 | 5 | 0 | 0 | 0.17 | 0.20 | 0.23 | 0.38 | 0.24 |
| uk_price_paid | real-estate-uk | 800×16 | ✓ | 1674 | 8 | 0 | 0 | 0.14 | 0.17 | 0.25 | 0.42 | 0.24 |
| bl_flickr_books | library | 800×15 | ✓ | 1780 | 6 | 1 | 0 | 0.18 | 0.28 | 0.22 | 0.42 | 0.28 |
| spotify | music | 800×23 | ✓ | 4675 | 3 | 1 | 0 | 0.20 | 0.28 | 0.30 | 0.36 | 0.28 |
| glassdoor_jobs | job-listings | 800×14 | ✓ | 1718 | 5 | 0 | 0 | 0.20 | 0.30 | 0.22 | 0.43 | 0.29 |
| ct_real_estate | real-estate-us | 800×14 | ✓ | 4550 | 6 | 0 | 0 | 0.23 | 0.29 | 0.24 | 0.40 | 0.29 |
| bx_books | books | 800×8 | ✓ | 1641 | 3 | 1 | 0 | 0.21 | 0.28 | 0.18 | 0.54 | 0.30 |
| worldcities | geography | 800×4 | ✓ | 914 | 2 | 0 | 0 | 0.41 | 0.11 | 0.22 | 0.69 | 0.36 |
| fec_indiv80 | political-finance | 800×21 | ✓ | 4855 | 3 | 2 | 0 | 0.40 | 0.34 | 0.48 | 0.86 | 0.52 |
| payroll_nyc | jobs | 800×17 | ✓ | 4587 | 3 | 2 | 0 | 0.45 | 0.56 | 0.42 | 0.73 | 0.54 |
| paris_trees | urban-fr | 800×16 | ✓ | 3309 | 5 | 1 | 0 | 0.43 | 0.54 | 0.55 | 0.72 | 0.56 |
| ev_wa | vehicles | 800×16 | ✓ | 4885 | 4 | 2 | 0 | 0.62 | 0.67 | 0.73 | 0.90 | 0.73 |
