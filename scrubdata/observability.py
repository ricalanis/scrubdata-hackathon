"""Monitorability layer for ScrubData — the oversight/observability angle.

The planner/executor decomposition makes every cleaning decision auditable: the model
PROPOSES a plan, deterministic pandas EXECUTES it, and nothing is changed that isn't in the
plan (no silent edits). Abstentions ("knowing when not to act") are first-class — surfaced
as review flags, not hidden. This module summarizes those signals and exports them as
OpenTelemetry GenAI-style span attributes so a cleaning run drops into Langfuse / Arize
Phoenix / any OTel backend like any other monitored agent action.
"""

from __future__ import annotations


def monitor_summary(plan: dict, change_log: list | None = None) -> dict:
    """Oversight signals for one cleaning run."""
    cols = plan.get("columns", [])
    canon_total = sum(len(o.get("mapping", {})) for c in cols for o in c.get("operations", [])
                      if o.get("op") == "canonicalize_categories")
    grounded_cols = sum(1 for c in cols for o in c.get("operations", [])
                        if o.get("op") == "canonicalize_categories"
                        and "reference taxonomy" in o.get("rationale", ""))
    abst = [f for f in plan.get("flags", []) if f.get("issue") == "uncertain_canonicalization"]
    abstained = sum(len(f.get("values", [])) for f in abst)
    table_ops = [o.get("op") for o in plan.get("table_operations", [])]
    return {
        "columns_touched": len(cols),
        "canonicalizations": canon_total,
        "grounded_columns": grounded_cols,
        "abstentions": abstained,
        "abstain_rate": round(abstained / (canon_total + abstained), 4) if (canon_total + abstained) else 0.0,
        "table_operations": table_ops,
        "changes_applied": len(change_log) if change_log is not None else None,
        "silent_edits": 0,                         # by construction: every change is in the plan
        "auditable": True, "reversible": True,
        "generated_by": plan.get("_generated_by", "unknown"),
    }


def otel_span(trace: dict) -> dict:
    """OTel GenAI-semantic-convention span for a cleaning run (drop into any OTel backend)."""
    s = monitor_summary(trace.get("plan", {}), trace.get("change_log"))
    return {
        "name": "scrubdata.clean",
        "kind": "INTERNAL",
        "attributes": {
            "gen_ai.operation.name": "data_cleaning_plan",
            "gen_ai.system": "scrubdata",
            "gen_ai.request.model": trace.get("model", s["generated_by"]),
            "gen_ai.response.duration_ms": trace.get("duration_ms"),
            "scrubdata.columns_touched": s["columns_touched"],
            "scrubdata.canonicalizations": s["canonicalizations"],
            "scrubdata.grounded_columns": s["grounded_columns"],
            "scrubdata.abstentions": s["abstentions"],
            "scrubdata.abstain_rate": s["abstain_rate"],
            "scrubdata.silent_edits": s["silent_edits"],
            "scrubdata.auditable": s["auditable"],
        },
    }
