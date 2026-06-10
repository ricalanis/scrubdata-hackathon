"""WS1 — selective prediction on the FULL planner (not just the retriever).

The grounded retriever already abstains; the planner's free canonicalization mappings on
non-grounded columns do not — they are where hospital precision dies (0.185). This module
scores EVERY mapping entry with deterministic, auditable evidence and keeps only entries
above a confidence threshold tau; the rest become review flags (abstention first-class).

Per-entry confidence combines three contract-preserving signals (no cell values emitted,
no gold access):
  * frequency support: the target exists in the column and dominates the source
    (errors are rare; canonicals are frequent — the cluster-repair prior)
  * variant similarity: the source is a string-variant of the target (edit similarity);
    alias resolution is handled upstream by the reference dictionary, so here a low-sim
    pair is an arbitrary rewrite, not a repair
  * reference agreement: for reference-typed columns, the retriever's verdict (its own
    calibrated confidence) overrides

Sweeping tau yields the precision-coverage curve (eval/precision_curve.py).
"""

from __future__ import annotations

import difflib
from collections import Counter

from . import detect


def _norm(s: str) -> str:
    return "".join(c.lower() for c in str(s) if c.isalnum())


def _letters(s: str) -> str:
    return "".join(c.lower() for c in str(s) if c.isalpha())


def entry_confidence(raw: str, canon: str, freq: Counter) -> float:
    """Deterministic confidence in mapping raw -> canon within a column.

    Three HARD gates (each kills a measured hospital failure class):
      * errors are rare: a value occurring >= 3 times is data, not a typo (de kalb x92)
      * repair to the dominant form only: the target must be a frequent column value,
        clearly dominating the source (yex -> yexu, a typo mapped to a worse typo)
      * code discipline: digit-bearing values (ak_hf-1) repair only when the letter
        part is near-identical (allows amix-2 -> ami-2, blocks ak_ -> al_)
    """
    n_raw, n_canon = freq.get(raw, 0), freq.get(canon, 0)
    nr, nc = _norm(raw), _norm(canon)
    if not nr or not nc:
        return 0.0
    if n_raw >= 3:                                       # frequent = legit data
        return 0.0
    if n_canon < max(2, 2 * max(n_raw, 1)):              # target must dominate
        return 0.0
    digits = sum(c.isdigit() for c in raw)
    if digits and digits >= 0.15 * len(raw):             # code-shaped value
        lr, lc = _letters(raw), _letters(canon)
        if not lr or difflib.SequenceMatcher(None, lr, lc).ratio() < 0.85:
            return 0.0
        if "".join(c for c in raw if c.isdigit()) != "".join(c for c in canon if c.isdigit()):
            return 0.0
    sim = 1.0 if nr == nc else difflib.SequenceMatcher(None, nr, nc).ratio()
    support = min(1.0, n_canon / (2.0 * max(n_raw, 1)))
    return round(sim * (0.5 + 0.5 * support), 4)


def verify_plan(df, plan: dict, tau: float = 0.6) -> dict:
    """Return a plan whose canonicalize mappings keep only entries with confidence >= tau.
    Dropped entries become review flags. Reference-grounded ops (rationale marks them)
    pass through untouched — the retriever already abstained for them."""
    import copy
    out = copy.deepcopy(plan)
    flags = out.setdefault("flags", [])
    for col in out.get("columns", []):
        name = col.get("name")
        if name not in getattr(df, "columns", []):
            continue
        values = [str(v).strip() for v in df[name].tolist() if not detect.is_missing(v)]
        freq = Counter(values)
        for op in col.get("operations", []):
            if op.get("op") != "canonicalize_categories":
                continue
            if "reference taxonomy" in op.get("rationale", ""):
                continue                      # grounded: retriever-calibrated already
            mapping = op.get("mapping", {})
            kept, dropped = {}, {}
            for raw, canon in mapping.items():
                conf = entry_confidence(str(raw), str(canon), freq)
                (kept if conf >= tau else dropped)[str(raw)] = (str(canon), conf)
            op["mapping"] = {r: c for r, (c, _) in kept.items()}
            op["_verified"] = {"tau": tau, "kept": len(kept), "dropped": len(dropped)}
            if dropped:
                flags.append({
                    "column": name, "issue": "low_confidence_canonicalization",
                    "values": list(dropped)[:20], "action": "left_for_review",
                    "rationale": f"{len(dropped)} proposed merge(s) below confidence "
                                 f"{tau} — left unchanged for review.",
                })
    out["columns"] = [c for c in out.get("columns", [])
                      if any(o.get("op") != "canonicalize_categories" or o.get("mapping")
                             for o in c.get("operations", []))]
    return out


def make_verified_planner(base_planner, tau: float = 0.6):
    """Wrap any planner with the per-entry verifier (selective prediction)."""
    def planner(df, *_):
        return verify_plan(df, base_planner(df), tau=tau)
    return planner
