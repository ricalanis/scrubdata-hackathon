"""Tier-1 PII detection + masking primitives — deterministic, zero-model, CPU-instant.

The trust story ("runs locally, nothing leaves your machine") only matters if sensitive
data is actually handled. Tier-1 types a column as PII from its DISTINCT values (the same
aggregation trick the profiler uses) with regex + CHECKSUM validators — Luhn for cards,
IBAN mod-97 — so the high-sensitivity calls are mathematically confirmed, not pattern-
guessed. A tier-2 small NER model (OpenMed-PII / GLiNER) can extend coverage later behind
the same contract: confident -> act, uncertain -> ABSTAIN and flag for review.

Masking transforms (used by the executor ops) are deterministic and format-preserving:
  mask:         keep the last 4 (cards/SSN/phones) or first char of an email local-part
  hash:         salted SHA-256 (salt carried in the plan so apply is replayable)
  pseudonymize: stable surrogate like EMAIL_3fa9c2 — same input -> same token, so
                joins/groupbys survive masking.
"""

from __future__ import annotations

import hashlib
import re

from . import detect

# --- value validators ---------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_PHONE_RE = re.compile(r"^\+?[\d(][\d\s().\-]{6,18}\d$")
_CARD_RE = re.compile(r"^[\d][\d \-]{11,21}\d$")
_IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")
_IPV4_RE = re.compile(
    r"^((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$")
_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")


def luhn_ok(digits: str) -> bool:
    """Luhn checksum — the validator that makes credit-card typing confirmed, not guessed."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_credit_card(v: str) -> bool:
    if not _CARD_RE.match(v):
        return False
    digits = re.sub(r"[ \-]", "", v)
    return 13 <= len(digits) <= 19 and digits.isdigit() and luhn_ok(digits)


def _is_iban(v: str) -> bool:
    s = v.replace(" ", "").upper()
    if not _IBAN_RE.match(s):
        return False
    # mod-97: move first 4 chars to the end, letters -> 10..35, number % 97 == 1
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


def _is_phone(v: str) -> bool:
    if _SSN_RE.match(v):          # specific beats generic: a dashed SSN is not a phone
        return False
    return bool(_PHONE_RE.match(v)) and 7 <= sum(c.isdigit() for c in v) <= 15


# ordered: checksum-confirmed first, then high-signal patterns. A validator may only
# claim a column if its hit_rate over distinct values clears the threshold.
VALIDATORS: list[tuple[str, bool, object]] = [
    ("credit_card", True, _is_credit_card),
    ("iban", True, _is_iban),
    ("ssn", False, lambda v: bool(_SSN_RE.match(v))),
    ("email", False, lambda v: bool(_EMAIL_RE.match(v))),
    ("ip_address", False, lambda v: bool(_IPV4_RE.match(v))),
    ("mac_address", False, lambda v: bool(_MAC_RE.match(v))),
    ("phone", False, _is_phone),
]

# auto-mask only these by default: checksum-confirmed or so sensitive that leaving them
# in a "cleaned" output would be a worse failure than over-masking. Contact columns
# (email/phone) are flagged, not auto-masked — users usually need them.
AUTO_MASK_TYPES = {"credit_card", "iban", "ssn"}


def detect_column_pii(name: str, values, max_distinct: int = 80,
                      threshold: float = 0.6) -> dict | None:
    """Type a column as PII from its distinct values. Returns
    {pii_type, confidence, tier, hit_rate, checksum} or None."""
    distinct = list(dict.fromkeys(
        str(v).strip() for v in values if not detect.is_missing(v)))[:max_distinct]
    if len(distinct) < 3:
        return None
    best = None
    for pii_type, checksum, fn in VALIDATORS:
        hits = sum(1 for v in distinct if fn(v))
        rate = hits / len(distinct)
        if rate >= threshold and (best is None or rate > best["hit_rate"]):
            best = {"pii_type": pii_type, "confidence": round(rate, 3), "tier": 1,
                    "hit_rate": round(rate, 3), "checksum": checksum}
    return best


# --- masking transforms (deterministic; used by executor ops) ------------------

def mask_value(v, pii_type: str = ""):
    """Format-preserving partial mask. Keeps just enough to recognize, never enough
    to reconstruct."""
    if detect.is_missing(v):
        return v
    s = str(v)
    if pii_type == "email" or "@" in s:
        local, _, domain = s.partition("@")
        return (local[:1] + "***@" + domain) if domain else "***"
    digits = [c for c in s if c.isdigit()]
    if len(digits) >= 7:                       # cards / SSNs / phones: keep last 4 digits
        keep = set()
        seen = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i].isdigit():
                seen += 1
                if seen <= 4:
                    keep.add(i)
        return "".join(c if (i in keep or not c.isalnum()) else "*" for i, c in enumerate(s))
    return s[:1] + "*" * max(len(s) - 1, 2)


# --- tier-2: small NER column typer (optional, lazy; OpenMed-PII 44M) ----------
#
# Transfer-validated (scripts/pii_transfer_check.py): 100% detection on BARE cell
# values for names/addresses — no context template needed. The model also correctly
# types city/county/occupation, but those are quasi-identifiers, not maskable PII, so
# tier-2 only acts on a SENSITIVE allowlist and requires a column-level coverage vote
# (same contract as reconcile.infer_reference_type).

SENSITIVE_NER_MAP = {
    "first_name": "person_name", "last_name": "person_name", "name": "person_name",
    "street_address": "address", "address": "address",
    "ssn": "ssn", "credit_card": "credit_card", "email": "email",
    "phone_number": "phone", "date_of_birth": "date_of_birth",
    "passport": "passport", "driver_license": "driver_license",
    "medical_record_number": "medical_record_number", "account_number": "account_number",
}

_NER_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_ner_pipe = None


def _get_ner():
    """Lazy singleton; None when transformers isn't installed (tier-2 silently off)."""
    global _ner_pipe
    if _ner_pipe is None:
        try:
            from transformers import pipeline
            _ner_pipe = pipeline("token-classification", model=_NER_MODEL,
                                 aggregation_strategy="simple")
        except Exception:  # noqa: BLE001
            _ner_pipe = False
    return _ner_pipe or None


def detect_column_pii_ner(name: str, values, max_distinct: int = 40,
                          min_coverage: float = 0.55) -> dict | None:
    """Tier-2: type a column as sensitive PII by NER-voting over sampled distinct
    values. Only sensitive-allowlisted entity types count; the column verdict needs
    majority coverage. Returns the same dict shape as detect_column_pii (tier=2)."""
    pipe = _get_ner()
    if pipe is None:
        return None
    from collections import Counter
    distinct = list(dict.fromkeys(
        str(v).strip() for v in values if not detect.is_missing(v)))[:max_distinct]
    if len(distinct) < 3:
        return None
    votes: Counter = Counter()
    for v in distinct:
        try:
            ents = pipe(v)
        except Exception:  # noqa: BLE001
            continue
        hit_types = {SENSITIVE_NER_MAP[e["entity_group"]] for e in ents
                     if e.get("entity_group") in SENSITIVE_NER_MAP and e.get("score", 0) >= 0.5}
        for t in hit_types:
            votes[t] += 1
    if not votes:
        return None
    ptype, n = votes.most_common(1)[0]
    coverage = n / len(distinct)
    if coverage < min_coverage:
        return None
    return {"pii_type": ptype, "confidence": round(coverage, 3), "tier": 2,
            "hit_rate": round(coverage, 3), "checksum": False}


def hash_value(v, salt: str):
    if detect.is_missing(v):
        return v
    return hashlib.sha256((salt + str(v).strip()).encode()).hexdigest()[:16]


def pseudonymize_value(v, salt: str, pii_type: str = "pii"):
    """Stable surrogate (EMAIL_3fa9c2): same input -> same token, joins survive."""
    if detect.is_missing(v):
        return v
    digest = hashlib.sha256((salt + str(v).strip()).encode()).hexdigest()[:6]
    return f"{pii_type.upper()}_{digest}"
