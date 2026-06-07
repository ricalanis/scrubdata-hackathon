"""Assemble one synthetic training example: clean table -> dirty table + plan.

make_example() returns a dict with the clean reference, the dirtied dataframe,
the ground-truth plan, and the dirty-data profile. build_dataset.py verifies each
by running the executor and keeps only perfectly-recovered examples.
"""

from __future__ import annotations

import random

import pandas as pd

from scrubdata.profiler import profile_dataframe

from .fields import ARCHETYPES


def _pick_columns(rng: random.Random, k: int):
    """Pick k archetypes (with replacement across types) and unique column names."""
    chosen, used_names = [], set()
    pool = list(ARCHETYPES)
    rng.shuffle(pool)
    for arch in pool:
        if len(chosen) >= k:
            break
        name = next((n for n in arch.names if n not in used_names), None)
        if name is None:
            continue
        used_names.add(name)
        chosen.append((name, arch))
    return chosen


def make_example(rng: random.Random) -> dict:
    n_rows = rng.randint(8, 18)
    n_cols = rng.randint(3, 6)
    cols = _pick_columns(rng, n_cols)

    clean_data: dict[str, list] = {}
    dirty_data: dict[str, list] = {}
    plan_columns: list[dict] = []

    for name, arch in cols:
        clean_vals = arch.gen_clean(rng, n_rows)
        dirty_vals, clean_vals, ops, issues = arch.corrupt(rng, clean_vals)
        clean_data[name] = clean_vals
        dirty_data[name] = dirty_vals
        if ops:
            plan_columns.append({
                "name": name,
                "detected_semantic_type": arch.semantic_type,
                "issues": issues,
                "operations": ops,
            })

    clean_df = pd.DataFrame(clean_data)
    # Drop any accidental duplicate clean rows so dedup verification is exact.
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    n_rows = len(clean_df)
    dirty_df = pd.DataFrame({c: v[:n_rows] for c, v in dirty_data.items()})

    # --- anomaly flags (flag-only: value is KEPT, not changed) ---
    # Teaches the model to surface implausible values without silently editing them.
    flags: list[dict] = []
    numeric_cols = [c["name"] for c in plan_columns
                    if any(o["op"] in ("parse_currency", "parse_number")
                           for o in c["operations"])]
    if numeric_cols and n_rows >= 3 and rng.random() < 0.4:
        col = rng.choice(numeric_cols)
        i = rng.randrange(n_rows)
        anomaly = rng.choice([9_999_999, -100, 0])
        dirty_df.at[i, col] = str(anomaly)
        clean_df.at[i, col] = float(anomaly)   # unchanged by flag-only execution
        flags.append({"column": col, "issue": "out_of_range", "action": "flag_only",
                      "rationale": f"Value {anomaly} is implausible for '{col}'; "
                                   f"flagged for human review, not auto-changed."})

    table_ops: list[dict] = []

    # --- table-level corruptions ---
    if rng.random() < 0.5:  # empty column
        empty_name = rng.choice(["notes2", "col_x", "extra", "unnamed"])
        dirty_df[empty_name] = ""
        table_ops.append({"op": "drop_empty_columns", "columns": [empty_name],
                          "rationale": "Dropped column(s) with no data."})

    extra_rows = []
    if rng.random() < 0.6 and n_rows >= 2:  # exact duplicate rows
        k = rng.randint(1, 2)
        dup_idx = rng.sample(range(n_rows), k)
        extra_rows.extend(dirty_df.iloc[dup_idx].to_dict("records"))
        table_ops.append({"op": "drop_exact_duplicates",
                          "rationale": f"Removed {k} exact duplicate row(s)."})
    if rng.random() < 0.4:  # an empty row
        empty = {c: "" for c in dirty_df.columns}
        extra_rows.append(empty)
        table_ops.append({"op": "drop_empty_rows", "rationale": "Removed 1 fully-empty row."})

    if extra_rows:
        dirty_df = pd.concat([dirty_df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Order table ops the way the executor expects (cols, rows, dedup).
    order = {"drop_empty_columns": 0, "drop_empty_rows": 1, "drop_exact_duplicates": 2}
    table_ops.sort(key=lambda o: order[o["op"]])

    profile = profile_dataframe(dirty_df)
    plan = {
        "dataset_summary": f"{len(dirty_df)} rows × {dirty_df.shape[1]} columns. "
                           f"{len(plan_columns)} column(s) need cleanup, "
                           f"{len(table_ops)} table-level fix(es).",
        "table_operations": table_ops,
        "columns": plan_columns,
        "flags": flags,
    }
    return {"clean_df": clean_df, "dirty_df": dirty_df, "plan": plan, "profile": profile}
