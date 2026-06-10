"""WS2 — pair-profiles: candidate-constrained canonicalization.

WS1 verifies the planner's OUTPUTS; WS2 constrains its INPUTS. The profiler emits, per
column, candidate (variant -> canonical) PAIRS backed by deterministic evidence — the
same contract-preserving signals the verifier scores (frequency support, edit
similarity, reference membership) — and the model planner SELECTS/ABSTAINS among the
listed candidates instead of free-generating mapping targets.

Two halves, both deterministic and auditable:
  * `candidate_pairs` / `pairs_for_df` — build the evidence-backed candidate sets.
  * `constrain_plan` — enforce the contract on the model's plan: any canonicalize
    mapping entry whose target is not a listed candidate for that surface is dropped
    to a review flag (the model may narrow the candidate set, never leave it).

Flag-gated OFF by default (env SCRUBDATA_PAIR_PROFILES=1): the v6 model was not
trained with candidate sections in the prompt, so this ships only if the zero-shot
measurement clears the WS1 gate (>=0.90 precision AND >0.413 coverage on hospital).
"""

from __future__ import annotations

import difflib
from collections import Counter

from . import detect
from .reconcile import default_index, infer_reference_type


def _norm(s: str) -> str:
    return "".join(c.lower() for c in str(s) if c.isalnum())


def candidate_pairs(values, idx=None, ctype=None, max_candidates: int = 3,
                    max_pairs: int = 40, min_sim: float = 0.6) -> list[dict]:
    """Evidence-backed repair candidates for one column's suspicious values.

    A surface is suspicious when it is RARE (errors are rare — freq < 3). Candidates:
      * frequency: a frequent column value (dominating the surface) within edit
        similarity `min_sim`
      * reference: the nearest entity of the column's reference type (score >= 0.7)
    Returns [{raw, count, candidates: [{canon, sim, support, source}]}], capped.
    """
    vals = [str(v).strip() for v in values if not detect.is_missing(v)]
    freq = Counter(vals)
    if ctype is None:
        idx = idx or default_index()
        ctype = infer_reference_type(vals, idx)
    frequent = [(v, n) for v, n in freq.most_common(60) if n >= 2]
    out = []
    for raw in freq:
        n_raw = freq[raw]
        if n_raw >= 3 or not _norm(raw):
            continue
        cands = []
        for canon, n_canon in frequent:
            if canon == raw or n_canon < max(2, 2 * n_raw):
                continue
            sim = difflib.SequenceMatcher(None, _norm(raw), _norm(canon)).ratio()
            if sim >= min_sim:
                cands.append({"canon": canon, "sim": round(sim, 3),
                              "support": n_canon, "source": "frequency"})
        if ctype and idx is not None:
            b = idx.best(raw, ctype)
            if b is not None and b[1] >= 0.7 and _norm(b[0]) != _norm(raw):
                cands.append({"canon": b[0], "sim": b[1], "support": freq.get(b[0], 0),
                              "source": f"reference:{ctype}"})
        if cands:
            cands.sort(key=lambda c: -c["sim"])
            out.append({"raw": raw, "count": n_raw, "candidates": cands[:max_candidates]})
    out.sort(key=lambda p: (-p["candidates"][0]["sim"], p["raw"]))
    return out[:max_pairs]


def pairs_for_df(df) -> dict:
    """Per-column candidate pairs for every plausible text column (compact prompt
    payload: candidates collapse to their surface strings)."""
    by_col = {}
    for col in df.columns:
        series = df[col]
        if series.dtype.kind not in "OUS":
            continue
        pairs = candidate_pairs(series.tolist())
        if pairs:
            by_col[col] = [{"raw": p["raw"],
                            "candidates": [c["canon"] for c in p["candidates"]]}
                           for p in pairs]
    return by_col


def constrain_plan(plan: dict, pairs_by_col: dict) -> dict:
    """Enforce the candidate contract on a plan: canonicalize entries must map a
    listed raw to one of ITS candidates; everything else becomes a review flag.
    Reference-grounded ops pass through (already constrained by the reference)."""
    import copy
    out = copy.deepcopy(plan)
    flags = out.setdefault("flags", [])
    for col in out.get("columns", []):
        allowed = {p["raw"]: set(p["candidates"]) for p in pairs_by_col.get(col.get("name"), [])}
        for op in col.get("operations", []):
            if op.get("op") != "canonicalize_categories":
                continue
            if "reference taxonomy" in op.get("rationale", ""):
                continue
            mapping = op.get("mapping", {})
            kept = {r: c for r, c in mapping.items() if c in allowed.get(str(r), ())}
            dropped = [r for r in mapping if r not in kept]
            op["mapping"] = kept
            if dropped:
                flags.append({
                    "column": col.get("name"), "issue": "outside_candidate_pairs",
                    "values": dropped[:20], "action": "left_for_review",
                    "rationale": f"{len(dropped)} mapping(s) not in the evidence-backed "
                                 "candidate set — left unchanged for review.",
                })
    out["columns"] = [c for c in out.get("columns", [])
                      if any(o.get("op") != "canonicalize_categories" or o.get("mapping")
                             for o in c.get("operations", []))]
    return out
