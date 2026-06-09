"""Grounded planner wrapper — the RACOON pattern for our pipeline.

Wraps ANY base planner (the fine-tuned model OR the heuristic) and grounds its
canonicalization for reference-typed columns (country/state/city): the model proposes a
plan, then for any column that reconciles to a reference taxonomy we REPLACE its
free-generated canonicalization with the deterministic, retrieval-grounded mapping and
surface low-confidence/ambiguous values as review flags. This is the structural fix for
the model's over-correction (v5 repair_prec 0.16): the model never invents a canonical for
a grounded column — the reference does, and the model handles everything else.
"""

from __future__ import annotations


def make_grounded_planner(base_planner):
    """Return a planner that runs `base_planner` then grounds reference-typed columns."""
    from scrubdata.reconcile import default_index, grounded_mapping, infer_reference_type

    def planner(df, *_args):
        plan = base_planner(df)
        if not isinstance(plan, dict):
            return plan
        idx = default_index()
        by_name = {c.get("name"): c for c in plan.get("columns", [])}
        flags = plan.setdefault("flags", [])
        grounded_cols = 0

        for name in df.columns:
            series = df[name]
            ref_type = infer_reference_type(series.tolist(), idx=idx)
            if ref_type is None:
                continue
            mapping, abstained = grounded_mapping(series.tolist(), ref_type, idx=idx)
            if not mapping and not abstained:
                continue
            col = by_name.get(name)
            if col is None:
                col = {"name": name, "detected_semantic_type": ref_type,
                       "issues": [], "operations": []}
                plan.setdefault("columns", []).append(col)
                by_name[name] = col
            # drop the model's free-generated canonicalization, install the grounded one
            col["operations"] = [o for o in col.get("operations", [])
                                 if o.get("op") != "canonicalize_categories"]
            if mapping:
                col["operations"].append({
                    "op": "canonicalize_categories", "mapping": mapping,
                    "rationale": f"Reconciled {len(mapping)} value(s) to the {ref_type} "
                                 f"reference taxonomy (grounded, not free-generated).",
                })
                grounded_cols += 1
            if abstained:
                flags.append({
                    "column": name, "issue": "uncertain_canonicalization",
                    "values": abstained[:20], "action": "left_for_review",
                    "rationale": f"{len(abstained)} {ref_type} value(s) look like typos but "
                                 f"did not confidently match the reference — left for review.",
                })

        # prune columns the grounding emptied out
        plan["columns"] = [c for c in plan.get("columns", []) if c.get("operations")]
        plan["_grounded_columns"] = grounded_cols
        return plan

    return planner
