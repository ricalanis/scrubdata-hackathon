"""PII column-typing eval on OOD data (Gretel pii-masking-en-v1 test split, Apache-2.0).

Deliberately NOT Nemotron-PII (OpenMed's training set — would be in-distribution). Builds
per-type columns from Gretel's labeled entity values and measures tier-1 column typing:

  * detection rate per PII type (column typed correctly)
  * false-positive rate on negative columns drawn from harvested real gov/GitHub data

Tier-1 is checksum/pattern-based, so synthetic values that fail real checksums (e.g.
non-Luhn card numbers) are expected misses — reported honestly, since rejecting
checksum-invalid "cards" is correct behavior for the validator tier.

    uv run --with pyarrow python -m eval.pii_slice
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scrubdata.pii import detect_column_pii, luhn_ok

# Gretel entity type -> our tier-1 pii_type (None = covered by tier-2/none, skip here)
TYPE_MAP = {
    "email": "email", "email_address": "email",
    "phone_number": "phone", "phone": "phone",
    "ssn": "ssn", "us_social_security_number": "ssn",
    "credit_card_number": "credit_card", "credit_card": "credit_card",
    "ipv4": "ip_address", "ip_address": "ip_address", "ipv6": None,
    "iban": "iban", "mac_address": "mac_address",
}

NEGATIVE_SOURCES = [  # (cache csv, column) — real non-PII categorical columns
    ("restaurants_nyc.csv", "cuisine_description"),
    ("restaurants_nyc.csv", "boro"),
    ("svc311_nyc.csv", "complaint_type"),
    ("biz_chicago.csv", "city"),
    ("film_nyc.csv", "category"),
    ("ev_wa.csv", "model"),
    ("spotify.csv", "playlist_genre"),
]


def load_gretel_columns(min_values: int = 30, cap: int = 80) -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("gretelai/gretel-pii-masking-en-v1",
                        "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    by_type: dict[str, list[str]] = defaultdict(list)
    for ents in df["entities"]:
        try:
            parsed = ast.literal_eval(ents) if isinstance(ents, str) else ents
        except (ValueError, SyntaxError):
            continue
        for e in parsed:
            types = e.get("types") or []
            val = str(e.get("entity", "")).strip()
            if not val:
                continue
            for t in types:
                ours = TYPE_MAP.get(str(t).lower())
                if ours:
                    by_type[ours].append(val)
    return {t: vals[:cap] for t, vals in by_type.items() if len(vals) >= min_values}


def negatives(nrows: int = 400) -> dict:
    out = {}
    cache = Path("data/real/cache")
    for fname, col in NEGATIVE_SOURCES:
        p = cache / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, nrows=nrows,
                             on_bad_lines="skip", encoding_errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if col in df.columns:
            out[f"{fname.split('.')[0]}:{col}"] = df[col].tolist()
    return out


def main() -> None:
    pos = load_gretel_columns()
    print(f"\n=== PII column typing on Gretel test (OOD; tier-1 validators) ===\n")
    print(f"{'PII type':<14}{'n values':>9}{'predicted':>14}{'correct':>9}")
    print("-" * 48)
    correct = total = 0
    for ptype, vals in sorted(pos.items()):
        r = detect_column_pii(ptype, vals)
        pred = r["pii_type"] if r else "(none)"
        ok = pred == ptype
        correct += ok; total += 1
        note = ""
        if ptype == "credit_card" and not ok:
            valid = sum(1 for v in vals if luhn_ok("".join(ch for ch in v if ch.isdigit()) or "0"))
            note = f"  ({valid}/{len(vals)} pass Luhn — synthetic numbers w/o valid checksums)"
        print(f"{ptype:<14}{len(vals):>9}{pred:>14}{str(ok):>9}{note}")
    print(f"\npositive column detection: {correct}/{total}")

    neg = negatives()
    fp = 0
    for name, vals in neg.items():
        r = detect_column_pii(name.split(":")[1], vals)
        if r:
            fp += 1
            print(f"  FALSE POSITIVE: {name} -> {r['pii_type']}")
    print(f"negative columns flagged: {fp}/{len(neg)} (false-positive rate "
          f"{fp / len(neg):.2f})" if neg else "no negatives found")


if __name__ == "__main__":
    main()
