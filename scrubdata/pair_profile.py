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
    frequent = [(v, _norm(v), n) for v, n in freq.most_common(60) if n >= 2]
    out = []
    # bounded work: full-table columns can carry tens of thousands of unique rare
    # surfaces (measured: 28k uniques -> ~25M SequenceMatcher calls -> Modal eval
    # timeout). Cap the rare values examined; cheap prefilters before any ratio().
    rare = [r for r in freq if freq[r] < 3 and _norm(r)]
    MAX_RARE = 4000
    if len(rare) > MAX_RARE:
        rare = rare[:MAX_RARE]
    for raw in rare:
        n_raw = freq[raw]
        nr = _norm(raw)
        cands = []
        for canon, nc, n_canon in frequent:
            if canon == raw or n_canon < max(2, 2 * n_raw):
                continue
            if not nc or nc[0] != nr[0] or abs(len(nc) - len(nr)) > max(2, len(nc) // 4):
                continue                              # cheap gates before ratio()
            m = difflib.SequenceMatcher(None, nr, nc)
            if m.real_quick_ratio() < min_sim or m.quick_ratio() < min_sim:
                continue
            sim = m.ratio()
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


def suspects_for_column(values, max_suspects: int = 25) -> list[dict]:
    """Profile-visibility section: rare anomalous surfaces with evidence-backed
    repair candidates, for EVERY text column INCLUDING high-cardinality ones (where
    value_counts truncation otherwise hides dirty cells from the planner entirely).

    Two trigger classes:
      * candidate-backed: a rare surface with a frequency-dominant / reference
        near-match (candidate_pairs — the WS2 machinery)
      * artifact carrier: a surface containing encoding/punctuation artifacts
        (unicode quotes/dashes/NBSP) — candidates may be empty; the planner's
        normalize_punctuation op or a review flag handles them

    Output shape (compact, bounded): [{raw, count, candidates: [str, ...]}]."""
    from . import detect

    out = candidate_pairs(values, max_pairs=max_suspects)
    listed = {p["raw"] for p in out}
    suspects = [{"raw": p["raw"], "count": p["count"],
                 "candidates": [c["canon"] for c in p["candidates"]]}
                for p in out]
    if len(suspects) < max_suspects:
        freq = Counter(str(v).strip() for v in values if not detect.is_missing(v))
        for raw, n in freq.items():
            if len(suspects) >= max_suspects:
                break
            if n < 3 and raw not in listed and detect.has_unicode_punctuation([raw]):
                suspects.append({"raw": raw, "count": n, "candidates": []})
    return suspects


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
