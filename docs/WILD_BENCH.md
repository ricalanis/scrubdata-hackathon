# Wild Bench — can the shipped system clean real-world tables?

Behavioral audit + seeded inject-recovery per dataset (eval/wild_bench.py).

| dataset | domain | rows×cols | valid | changes | flags | PII | silent | typo | ocr | case | ws | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| schools_nyc | education | 800×41 | ✓ | 14385 | 1 | 5 | 0 | 0.09 | 0.14 | 0.13 | 0.22 | 0.15 |
