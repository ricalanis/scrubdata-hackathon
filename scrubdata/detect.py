"""Shared semantic-type detection heuristics.

Deterministic, regex/rule based. The fine-tuned model will eventually *confirm
and extend* these (especially fuzzy categorical canonicalization), but the
profiler needs cheap first-pass signals to put in front of the model.
"""

from __future__ import annotations

import re

# Tokens that mean "missing" but aren't a real blank (PRODUCT.md 2.C).
DISGUISED_NULLS = {
    "", "n/a", "na", "n.a.", "-", "--", "—", "null", "none", "nan",
    "#n/a", "tbd", "?", "unknown", "missing",
    # literal-string nulls seen in real tables (e.g. Raha hospital uses "empty")
    "empty", "(empty)",
}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[+(]?[\d][\d\s().\-]{6,}$")
_CURRENCY_RE = re.compile(r"^\s*[($]?\s*-?[\d][\d.,]*\s*\)?\s*$")
_PERCENT_RE = re.compile(r"^\s*-?[\d][\d.,]*\s*%\s*$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_SLASH_DATE_RE = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")
_TEXT_DATE_RE = re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}$|^[A-Za-z]{3,9}[\s\-]\d{2,4}$")
_EXCEL_SERIAL_RE = re.compile(r"^\d{4,5}$")

BOOL_TRUE = {"yes", "y", "true", "t", "1", "✓"}
BOOL_FALSE = {"no", "n", "false", "f", "0"}

# Tiny built-in canonical dictionaries. The MOCK planner uses these; the model
# will generate richer/context-specific mappings. Kept intentionally small.
COUNTRY_CANON = {
    "usa": "United States", "u.s.a.": "United States", "u.s.a": "United States",
    "us": "United States", "u.s.": "United States", "united states": "United States",
    "united states of america": "United States", "america": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "u.k": "United Kingdom",
    "united kingdom": "United Kingdom", "great britain": "United Kingdom",
    "canada": "Canada", "germany": "Germany",
}


def normalize_token(v) -> str:
    return str(v).strip().lower()


def is_missing(v) -> bool:
    if v is None:
        return True
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    return normalize_token(v) in DISGUISED_NULLS


def _frac(values, pred) -> float:
    vals = [v for v in values if not is_missing(v)]
    if not vals:
        return 0.0
    return sum(1 for v in vals if pred(str(v).strip())) / len(vals)


def detect_semantic_type(name: str, values) -> str:
    """Return a coarse semantic type for a column from its non-missing values."""
    vals = [v for v in values if not is_missing(v)]
    if not vals:
        return "empty"

    lname = name.lower()

    def frac(pred):
        return _frac(vals, pred)

    # High-precision regex signals first.
    if frac(lambda s: bool(_EMAIL_RE.match(s))) > 0.7:
        return "email"
    if frac(lambda s: bool(_PERCENT_RE.match(s))) > 0.6:
        return "percent"
    # Booleans: small distinct set drawn from the bool vocab.
    distinct = {normalize_token(v) for v in vals}
    if distinct and distinct <= (BOOL_TRUE | BOOL_FALSE):
        return "boolean"
    if "zip" in lname or "postal" in lname or "zcta" in lname:
        return "text"            # ZIP/ZIP+4/ZCTA: never phone, date or number (leading
        #                          zeros + dashes are data, not formatting to fix)
    if frac(lambda s: bool(_PHONE_RE.match(s)) and sum(c.isdigit() for c in s) >= 7) > 0.6 \
            and ("phone" in lname or "tel" in lname or frac(lambda s: any(c in s for c in "()+-")) > 0.3):
        return "phone"
    if frac(_looks_like_date) > 0.6:
        # the Excel-serial branch alone (5-digit ints) is weak evidence — a ZIP/ID
        # column in the 36000-50000 range types as 2010s dates (measured: 380 damaged
        # cells). Serial-only columns need a date-ish NAME to qualify.
        if frac(lambda s: bool(_EXCEL_SERIAL_RE.match(s.strip()))) > 0.9 \
                and not any(t in lname for t in ("date", "time", "day", "created",
                                                 "updated", "modified", "dob")):
            return "number"
        return "date"
    if frac(lambda s: bool(_CURRENCY_RE.match(s))) > 0.6:
        # Distinguish currency (has $ or thousands grouping) from plain number.
        if frac(lambda s: "$" in s or "(" in s) > 0.1 or "amount" in lname or "price" in lname \
                or "revenue" in lname or "cost" in lname:
            return "currency"
        return "number"

    # Country / categorical fallbacks.
    if "country" in lname or all(normalize_token(v) in COUNTRY_CANON for v in vals):
        if frac(lambda s: normalize_token(s) in COUNTRY_CANON) > 0.5:
            return "country"
    # Low cardinality => categorical.
    if len(distinct) <= max(10, int(0.3 * len(vals))) and len(distinct) < len(vals):
        return "categorical"
    return "text"


def _looks_like_date(s: str) -> bool:
    s = s.strip()
    if _ISO_DATE_RE.match(s) or _SLASH_DATE_RE.match(s) or _TEXT_DATE_RE.match(s):
        return True
    # Excel serial in a plausible modern range (2000-01-01 .. 2035).
    if _EXCEL_SERIAL_RE.match(s) and 36000 <= int(s) <= 50000:
        return True
    return False


def phone_shape(s) -> str:
    """Structural shape of a phone string (digits→D), e.g. '(DDD) DDD-DDDD'."""
    return re.sub(r"\d", "D", str(s).strip())


def date_formats_consistent(values) -> bool:
    """True if all non-missing date strings share one structural CONVENTION — an
    already-consistent column should NOT be re-formatted (convention-conservatism:
    measured ~2k damaged cells from ISO-converting uniformly M/D/YYYY columns).
    Digit runs are collapsed so '1/4/2016' and '12/23/2015' count as the same shape;
    a column is 'consistent' when one shape covers >=90% of values (the minority is
    typically the ERRORS, which are repair targets — not a license to re-format the
    whole column)."""
    shapes = [re.sub(r"\d+", "D", str(v).strip()) for v in values if not is_missing(v)]
    if not shapes:
        return True
    from collections import Counter
    return Counter(shapes).most_common(1)[0][1] / len(shapes) >= 0.9


def percent_formats_consistent(values) -> bool:
    """True if every non-missing value carries the % suffix (uniform convention)."""
    vals = [str(v).strip() for v in values if not is_missing(v)]
    return bool(vals) and all(v.endswith("%") for v in vals)


def phone_formats_consistent(values) -> bool:
    """True if all non-missing phone values already share one format (don't reformat)."""
    shapes = {phone_shape(v) for v in values if not is_missing(v)}
    return len(shapes) <= 1


def has_whitespace_issues(values) -> bool:
    for v in values:
        if is_missing(v):
            continue
        s = str(v)
        if s != s.strip() or "  " in s:
            return True
    return False


_UNICODE_PUNCT = set(
    "\u2018\u2019\u201a\u2032\u00b4"   # curly/prime/acute single quotes
    "\u201c\u201d\u201e\u2033"          # curly double quotes
    "\u2013\u2014\u2012\u2015\u2212"   # en/em/figure/h-bar/minus dashes
    "\u00a0\u2009\u202f"                 # NBSP / thin / narrow no-break
    "\u200b\u200c\u200d\ufeff"          # zero-width characters
    "\u2026"                               # ellipsis
)


_MOJIBAKE_SIGNS = ("Ã", "â€", "�", "ï»¿")


def has_mojibake(values) -> bool:
    """True if any value shows UTF-8-as-cp1252 mis-decoding artifacts."""
    for v in values:
        if is_missing(v):
            continue
        s = str(v)
        if any(m in s for m in _MOJIBAKE_SIGNS):
            return True
    return False


def has_unicode_punctuation(values) -> bool:
    """True if any value carries unicode punctuation artifacts (curly quotes, long
    dashes, NBSP, zero-width chars) — normalizable deterministically to ASCII."""
    for v in values:
        if is_missing(v):
            continue
        if any(c in _UNICODE_PUNCT for c in str(v)):
            return True
    return False


def casing_variants(values) -> bool:
    """True if the same token appears with different casing (=> needs casing fix)."""
    seen: dict[str, set] = {}
    for v in values:
        if is_missing(v):
            continue
        s = str(v).strip()
        seen.setdefault(s.lower(), set()).add(s)
    return any(len(variants) > 1 for variants in seen.values())
