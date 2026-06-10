"""PII leak-rate experiment: after the planner masks a table, does ANY detectable PII
survive in the output? The executable 'did masking actually work' check.

Builds tables of synthetic-but-valid PII (Luhn-valid cards, mod-97-valid IBANs, SSNs,
emails, phones) mixed with non-PII columns, runs the full profile->plan->execute
pipeline, then re-runs every tier-1 validator over the cleaned output. leak rate =
fraction of masked-policy cells still validating as PII. Deterministic (seeded).

    uv run python -m eval.pii_leak
"""

from __future__ import annotations

import random

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.pii import VALIDATORS, AUTO_MASK_TYPES, luhn_ok
from scrubdata.planner import mock_plan


def _make_card(rng: random.Random) -> str:
    digits = [4] + [rng.randint(0, 9) for _ in range(14)]
    # choose Luhn check digit
    for d in range(10):
        if luhn_ok("".join(map(str, digits + [d]))):
            return "".join(map(str, digits + [d]))
    return "".join(map(str, digits + [0]))


def _make_iban(rng: random.Random) -> str:
    bban = "".join(str(rng.randint(0, 9)) for _ in range(18))
    for chk in range(2, 99):
        cand = f"DE{chk:02d}{bban}"
        re = cand[4:] + cand[:4]
        if int("".join(str(int(c, 36)) for c in re)) % 97 == 1:
            return cand
    return f"DE00{bban}"


def build_table(n: int = 120, seed: int = 9) -> pd.DataFrame:
    rng = random.Random(seed)
    return pd.DataFrame({
        "card": [_make_card(rng) for _ in range(n)],
        "iban": [_make_iban(rng) for _ in range(n)],
        "ssn": [f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"
                for _ in range(n)],
        "email": [f"user{rng.randint(1,999)}@mail{rng.randint(1,9)}.com" for _ in range(n)],
        "city": [rng.choice(["Boston", "Chicago", "Dallas", "Phoenix"]) for _ in range(n)],
    })


def leak_rate(df: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    fns = {name: fn for name, _chk, fn in VALIDATORS}
    out = {}
    for col in df.columns:
        vals = [str(v) for v in cleaned[col].tolist()]
        leaks = {name: sum(1 for v in vals if fn(v)) for name, fn in fns.items()}
        out[col] = {k: v for k, v in leaks.items() if v}
    return out


def main() -> None:
    df = build_table()
    plan = mock_plan(df)
    cleaned, _ = apply_plan(df, plan)
    leaks = leak_rate(df, cleaned)
    masked_cols = [c["name"] for c in plan["columns"]
                   for o in c["operations"] if o["op"] == "mask_pii"]
    n_masked_cells = sum(len(df) for c in masked_cols)
    n_leaked = sum(v for col in masked_cols for v in leaks.get(col, {}).values())
    print(f"\n=== PII leak-rate ({len(df)} rows; masked columns: {masked_cols}) ===")
    for col in df.columns:
        print(f"  {col:>6}: residual detections = {leaks.get(col) or 'none'}")
    rate = n_leaked / n_masked_cells if n_masked_cells else 0.0
    print(f"\nLEAK RATE over masked cells: {n_leaked}/{n_masked_cells} = {rate:.4f}")
    print("(email is flag-only by policy; its residual detections are by design)")


if __name__ == "__main__":
    main()
