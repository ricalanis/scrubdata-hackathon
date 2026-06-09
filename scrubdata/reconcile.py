"""Reference-taxonomy reconciliation for grounded canonicalization.

The arXiv verdict ([[taxonomy-grounding]]): never free-generate or frequency-cluster a
canonical form — RECONCILE the dirty value against a type-scoped reference taxonomy and
ABSTAIN when nothing matches confidently. This is the structural fix for the
`guntxrsvillx -> huntsville` wrong-merge: we only map to a canonical that actually EXISTS
in the reference, and we decline otherwise.

A `ReferenceIndex` holds, per concept type, the canonical entities + their aliases, and
`reconcile(value, type)` returns (canonical, confidence) or None (ABSTAIN). Matching is
exact/alias first, then fuzzy (normalized edit-similarity) against the type-scoped set.

Default index is built from `pycountry` (countries + US states) — tiny, ships locally, no
fetch. Cities/orgs (GeoNames/ROR) can be registered via `add()` from a cached subset.
"""

from __future__ import annotations

import difflib
import functools


def _norm(s) -> str:
    return "".join(c.lower() for c in str(s) if c.isalnum())


class ReferenceIndex:
    """Type-scoped alias->canonical reference with exact + fuzzy reconciliation."""

    def __init__(self) -> None:
        self._exact: dict[str, dict[str, str]] = {}          # type -> {norm_surface: canonical}
        # type -> {first_char: [(canonical, norm_canonical)]} for fast fuzzy scan
        self._buckets: dict[str, dict[str, list[tuple[str, str]]]] = {}
        # fuzzy scans over big buckets (196k cities) are the eval bottleneck and the
        # same values recur constantly across datasets/seeds — memoize per (type, value)
        self._cache: dict[tuple[str, str], tuple | None] = {}

    def add(self, ctype: str, canonical: str, aliases=()) -> None:
        exact = self._exact.setdefault(ctype, {})
        exact[_norm(canonical)] = canonical
        for a in aliases:
            if _norm(a):
                exact.setdefault(_norm(a), canonical)
        nc = _norm(canonical)
        if nc:
            self._buckets.setdefault(ctype, {}).setdefault(nc[0], []).append((canonical, nc))

    def has_type(self, ctype: str) -> bool:
        return ctype in self._buckets

    def best(self, value, ctype: str):
        """Nearest reference canonical (canonical, score, margin) or None. `margin` =
        score - second-best score; a small margin means an AMBIGUOUS match (e.g. boxz
        ~ Box and ~ Boaz) that should abstain. Exact/alias = (canon, 1.0, 1.0)."""
        nv = _norm(value)
        if not nv or ctype not in self._buckets:
            return None
        key = (ctype, nv)
        if key in self._cache:
            return self._cache[key]
        hit = self._exact.get(ctype, {}).get(nv)
        if hit is not None:
            self._cache[key] = (hit, 1.0, 1.0)
            return self._cache[key]
        best, best_r, second_r = None, 0.0, 0.0
        for canonical, nc in self._buckets[ctype].get(nv[0], []):
            if abs(len(nc) - len(nv)) > 1 + len(nv) // 3:        # length prefilter
                continue
            r = difflib.SequenceMatcher(None, nv, nc).ratio()
            if r > best_r:
                best, best_r, second_r = canonical, r, best_r
            elif r > second_r:
                second_r = r
        out = None if best is None else (best, round(best_r, 3), round(best_r - second_r, 3))
        self._cache[key] = out
        return out

    def reconcile(self, value, ctype: str, threshold: float = 0.84):
        """Return (canonical, confidence) or None (ABSTAIN) — `best` gated by threshold."""
        b = self.best(value, ctype)
        return (b[0], b[1]) if (b is not None and b[1] >= threshold) else None


def infer_reference_type(values, idx: ReferenceIndex | None = None,
                         min_coverage: float = 0.55, sample: int = 80):
    """Type a column by the reference itself (research's column-typing step): if most
    distinct values reconcile to a given taxonomy, that's the column's concept type.
    Returns 'country'|'state'|'city' or None."""
    idx = idx or default_index()
    distinct = list(dict.fromkeys(str(x).strip() for x in values if str(x).strip()))[:sample]
    if len(distinct) < 4:
        return None
    best_type, best_cov = None, 0.0
    for ctype in ("country", "state", "city"):
        if not idx.has_type(ctype):
            continue
        hits = sum(1 for v in distinct if (b := idx.best(v, ctype)) and b[1] >= 0.80)
        cov = hits / len(distinct)
        if cov > best_cov:
            best_type, best_cov = ctype, cov
    return best_type if best_cov >= min_coverage else None


def _column_case(values) -> str:
    from collections import Counter
    styles: Counter = Counter()
    for v in values:
        s = str(v).strip()
        if not s or not s[:1].isalpha():
            continue
        styles["lower" if s.islower() else "upper" if s.isupper()
               else "title" if s.istitle() else "other"] += 1
    top = styles.most_common(1)
    return top[0][0] if top and top[0][0] != "other" else "title"


def _apply_case(s: str, style: str) -> str:
    return s.lower() if style == "lower" else s.upper() if style == "upper" else s


def grounded_mapping(values, ctype: str, idx: ReferenceIndex | None = None,
                     threshold: float = 0.84, review_floor: float = 0.70,
                     min_margin: float = 0.03, case_match: bool = True):
    """Ground a column's canonicalization in the type's reference taxonomy.

    Returns (mapping, abstained): `mapping` only contains dirty->canonical where the
    dirty value confidently AND UNAMBIGUOUSLY reconciles to a REAL reference entity (the
    structural fix for wrong-merges); the canonical is cast to the column's case
    convention. `abstained` = near-miss / ambiguous values surfaced for human review."""
    idx = idx or default_index()
    if not idx.has_type(ctype):
        return {}, []
    style = _column_case(values)
    mapping: dict[str, str] = {}
    abstained: list[str] = []
    for v in dict.fromkeys(str(x).strip() for x in values if str(x).strip()):
        b = idx.best(v, ctype)
        if b is None:
            continue
        canon, score, margin = b
        if score >= threshold and margin >= min_margin:
            cased = _apply_case(canon, style) if case_match else canon
            if cased != v:
                mapping[v] = cased
        elif score >= review_floor:        # near-miss or ambiguous -> ABSTAIN for review
            abstained.append(v)
    return mapping, abstained


@functools.lru_cache(maxsize=1)
def default_index() -> ReferenceIndex:
    """Countries + US states from pycountry (built-in, no fetch). Register cities/orgs
    separately from a cached GeoNames/ROR subset when available."""
    idx = ReferenceIndex()
    try:
        import pycountry
    except Exception:  # noqa: BLE001
        return idx
    _COMMON = {
        "United States": ["USA", "U.S.A.", "U.S.", "US", "America", "United States of America"],
        "United Kingdom": ["UK", "U.K.", "Britain", "Great Britain", "England"],
        "United Arab Emirates": ["UAE"], "South Korea": ["Korea", "Republic of Korea"],
        "Russian Federation": ["Russia"], "Czechia": ["Czech Republic"],
        "Netherlands": ["Holland", "The Netherlands"], "Viet Nam": ["Vietnam"],
    }
    for c in pycountry.countries:
        aliases = {c.alpha_2, c.alpha_3}
        official = getattr(c, "official_name", None)
        if official and official != c.name:
            aliases.add(official)
        aliases.update(_COMMON.get(c.name, []))
        idx.add("country", c.name, [a for a in aliases if a])
    for s in pycountry.subdivisions.get(country_code="US") or []:
        idx.add("state", s.name, [s.code.split("-")[-1]])
    # cities from the bundled reference (world-cities subset); swappable for a bigger
    # gazetteer (GeoNames cities500) — the reference's coverage is the accuracy ceiling.
    import os
    cities = os.path.join(os.path.dirname(__file__), "refdata", "cities.txt")
    if os.path.exists(cities):
        with open(cities, encoding="utf-8") as fh:
            for line in fh:
                name = line.strip()
                if name:
                    idx.add("city", name)
    return idx
