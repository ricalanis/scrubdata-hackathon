"""OpenRefine-style clustering baselines — the actual tool ScrubData competes with.

OpenRefine's two default clustering methods, as planner functions (so they slot into the
same eval as our grounded planner):

  * fingerprint (key collision): normalize -> ASCII-fold -> drop punctuation -> sort+dedupe
    tokens -> canonical = most frequent member of each fingerprint group. Catches case /
    whitespace / word-order / punctuation variants, but NOT typos or aliases.
  * nearest-neighbor (kNN / edit-distance): greedily merge a rarer value into a more-
    frequent one within an edit-similarity radius. Catches typos — but with NO reference
    it wrong-merges (guntxrsvillx -> huntsville), exactly the failure our grounding fixes.

These let us report the money result: grounded reconciliation vs the tool people actually
use, on the same wide validation suite.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter


def fingerprint(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(sorted(set(s.split())))


def _freq(values):
    return Counter(str(v).strip() for v in values if str(v).strip())


def fingerprint_clusters(values) -> dict:
    freq = _freq(values)
    groups: dict[str, list[str]] = {}
    for v in freq:
        groups.setdefault(fingerprint(v), []).append(v)
    mapping = {}
    for members in groups.values():
        if len(members) <= 1:
            continue
        canon = max(members, key=lambda m: freq[m])
        for m in members:
            if m != canon:
                mapping[m] = canon
    return mapping


def _norm(s: str) -> str:
    return "".join(c.lower() for c in str(s) if c.isalnum())


def knn_clusters(values, threshold: float = 0.82) -> dict:
    """OpenRefine nearest-neighbor: greedily attach a rarer value to a more-frequent
    canonical within edit-similarity `threshold`. NO reference -> over-merges."""
    freq = _freq(values)
    distinct = sorted(freq, key=lambda v: -freq[v])
    canon: list[tuple[str, str]] = []        # (value, normalized)
    mapping = {}
    for v in distinct:
        nv = _norm(v)
        if not nv:
            continue
        match = None
        for cval, cn in canon:
            if freq[cval] >= freq[v] and difflib.SequenceMatcher(None, nv, cn).ratio() >= threshold:
                match = cval
                break
        if match is not None and match != v:
            mapping[v] = match
        else:
            canon.append((v, nv))
    return mapping


def _plan(df, cluster_fn, tag: str) -> dict:
    columns = []
    for col in df.columns:
        mapping = cluster_fn(df[col].tolist())
        if mapping:
            columns.append({"name": col, "detected_semantic_type": "categorical",
                            "issues": [], "operations": [{
                                "op": "canonicalize_categories", "mapping": mapping,
                                "rationale": f"OpenRefine {tag} clustering."}]})
    return {"dataset_summary": f"OpenRefine {tag} baseline.", "table_operations": [],
            "columns": columns, "flags": [], "_generated_by": f"openrefine_{tag}"}


def openrefine_fingerprint_plan(df, profile=None) -> dict:
    return _plan(df, fingerprint_clusters, "fingerprint")


def openrefine_knn_plan(df, profile=None) -> dict:
    return _plan(df, knn_clusters, "knn")
