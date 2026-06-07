"""Training-data tooling for the ScrubData planner.

Synthetic-dirtying pipeline: generate CLEAN tables with known schemas, inject
controlled mess, and emit (dirty profile -> ground-truth plan) SFT pairs. Because
we created the mess, the ground-truth plan is known; every example is then
VERIFIED by running scrubdata.executor (dirty + plan should recover the clean
original), so the dataset is guaranteed-correct.
"""
