"""Evaluation harness for the ScrubData planner.

Measures any planner (`callable(dirty_df) -> plan dict`) against a held-out gold set:
- JSON-schema validity of the plan
- operation-level micro-F1 vs the gold plan
- canonicalization mapping micro-F1 (the fuzzy skill rules can't do)
- end-to-end cell-recovery (executor(dirty, plan) vs known-clean reference)

Two reference systems frame every run:
- HEURISTIC (`scrubdata.mock_plan`) = the baseline a fine-tuned model must beat.
- ORACLE (the gold plan itself) = the goalpost ceiling (~100% by construction).
"""
