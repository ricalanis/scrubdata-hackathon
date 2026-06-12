"""R2 verification — are the bit-identical devstral-24B / gemma-31B rows in
eval/results/scaling_arm.json a scoring collision (same applied edits) or a bug?

Loads both captured raw plans, pushes each through the EXACT scaling_arm.py
protocol (verify_plan tau=0.5 -> gated | union(mock_plan) -> _repairs_only ->
apply_plan), applies the final plans to the hospital dirty table, and diffs the
actual changed-cell sets {(row, col, old, new)}.

    uv run python -m eval.verify_scaling_coincidence
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan
from scrubdata.verifier import union_plans, verify_plan

from .precision_curve import _repairs_only
from .metrics import _cell_equal
from .run_real import _ensure_data, _load
from .run_real_multi import score as _cn_score
from .scaling_arm import kept_dropped

RESULTS = Path(__file__).resolve().parent / "results"
PLANS = {
    "devstral-small-2:24b-cloud":
        RESULTS / "scaling_devstral-small-2_24b-cloud_hospital_raw_plan.json",
    "gemma4:31b-cloud":
        RESULTS / "scaling_gemma4_31b-cloud_hospital_raw_plan.json",
}
TAU = 0.5


def edit_set(dirty, plan) -> set[tuple[int, str, str, str]]:
    """Apply a final plan and return the set of actually-changed cells."""
    cleaned, _ = apply_plan(dirty, _repairs_only(plan))
    edits = set()
    for j, col in enumerate(dirty.columns):
        if col not in cleaned.columns:
            continue
        for i in range(min(len(dirty), len(cleaned))):
            dv, ov = dirty.iat[i, j], cleaned.iloc[i][col]
            if str(dv) != str(ov):
                edits.add((i, col, str(dv), str(ov)))
    return edits


def main() -> None:
    t0 = time.time()
    _ensure_data()
    dirty, clean = _load()

    out: dict = {"task": "R2 — devstral/gemma identical-row verification",
                 "tau": TAU, "models": {}}
    per_model = {}
    for model, path in PLANS.items():
        raw = json.load(open(path))
        md5 = hashlib.md5(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        verified = verify_plan(dirty, raw, tau=TAU)
        k, d = kept_dropped(verified)
        union = union_plans(verified, mock_plan(dirty))
        sets = {"gated": edit_set(dirty, verified),
                "union": edit_set(dirty, union)}
        metrics = {}
        for label, plan in (("gated", verified), ("union", union)):
            cleaned, _ = apply_plan(dirty, _repairs_only(plan))
            m = _cn_score(dirty, clean, cleaned)
            metrics[label] = {"prec": round(m["precision"], 3),
                              "cov": round(m["recall"], 3),
                              "changed": m["_changed"], "fixed": m["_fixed"]}
        per_model[model] = {"sets": sets}
        out["models"][model] = {"raw_plan": str(path.name), "raw_plan_md5": md5,
                                "verifier_kept": k, "verifier_dropped": d,
                                "metrics": metrics}
        print(f"{model}: md5={md5} kept/dropped={k}/{d} "
              f"gated |edits|={len(sets['gated'])} union |edits|={len(sets['union'])}")

    def judge(edits):
        """Annotate each edit with correctness vs the gold clean table."""
        rows = []
        for i, col, old, new in sorted(edits):
            cv = clean.iat[i, list(dirty.columns).index(col)]
            rows.append({"row": i, "col": col, "old": old, "new": new,
                         "gold": cv, "correct_fix": _cell_equal(new, cv),
                         "was_error": not _cell_equal(old, cv)})
        return rows

    (a_name, a), (b_name, b) = per_model.items()
    out["diff"] = {}
    for label in ("gated", "union"):
        A, B = a["sets"][label], b["sets"][label]
        sym = A ^ B
        identical = not sym
        only_a, only_b = judge(A - B), judge(B - A)
        out["diff"][label] = {
            "|A|": len(A), "|B|": len(B),
            "|A∩B|": len(A & B), "|A∆B|": len(sym),
            "identical": identical,
            "only_A": only_a, "only_B": only_b,
            "only_A_correct_fixes": sum(e["correct_fix"] for e in only_a),
            "only_B_correct_fixes": sum(e["correct_fix"] for e in only_b),
        }
        print(f"{label}: |A|={len(A)} |B|={len(B)} |A∩B|={len(A & B)} "
              f"|A∆B|={len(sym)} identical={identical}")

    out["A"] = a_name
    out["B"] = b_name
    out["raw_plans_identical"] = (
        out["models"][a_name]["raw_plan_md5"] == out["models"][b_name]["raw_plan_md5"])
    if all(d["identical"] for d in out["diff"].values()):
        out["verdict"] = ("identical applied edit sets (same final repairs from "
                          "different raw plans)")
    else:
        # collision iff every differing edit on both sides is a correct fix of a
        # real error AND counts match — then changed/fixed (hence prec/cov) tie
        # exactly without the cell sets being equal.
        collision = all(
            d["|A|"] == d["|B|"]
            and d["only_A_correct_fixes"] == len(d["only_A"])
            and d["only_B_correct_fixes"] == len(d["only_B"])
            and len(d["only_A"]) == len(d["only_B"])
            for d in out["diff"].values())
        out["verdict"] = (
            "scoring collision — NOT identical edits: counts (changed, fixed) tie "
            "exactly because each model's unique edits are equal in number and all "
            "correct fixes of real errors; no harness bug"
            if collision else
            "edit sets differ and counts do not decompose as a clean collision — "
            "investigate harness")
    out["runtime_s"] = round(time.time() - t0, 1)

    dest = RESULTS / "scaling_coincidence.json"
    json.dump(out, open(dest, "w"), indent=1, ensure_ascii=False)
    print(f"\nverdict: {out['verdict']}")
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
