"""Seeded, self-verifying error injection — turns any CLEAN table into dirty/clean
validation. This is the de-biasing core of the north-star: our 20+ harvested clean
domains become per-cell-ground-truth validation across error types, far beyond any one
published benchmark.

Self-contained (no nlpaug/BART deps): we inject a KNOWN corruption into a clean cell, so
the (dirty -> clean) ground truth is exact and the run is reproducible (fixed seed).

Injects RECOVERABLE error types (the cleaner can restore the clean value): typo, ocr,
case, whitespace — i.e. the canonicalization + format axes. Targets CATEGORICAL text
columns (recurring values), where canonicalization is the task.
"""

from __future__ import annotations

import random
import string

_OCR = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5",
        "B": "8", "Z": "2", "z": "2", "g": "9", "G": "6", "b": "6"}


def _typo(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    if not s[i].isalpha():
        return s
    m = rng.random()
    if m < 0.55:                                  # substitute (the classic 'birminghxm')
        pool = string.ascii_uppercase if s[i].isupper() else string.ascii_lowercase
        return s[:i] + rng.choice([c for c in pool if c != s[i].lower()]) + s[i + 1:]
    if m < 0.78:                                  # delete
        return s[:i] + s[i + 1:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]    # transpose


def _ocr(s: str, rng: random.Random) -> str:
    idxs = [i for i, c in enumerate(s) if c in _OCR]
    if not idxs:
        return _typo(s, rng)
    i = rng.choice(idxs)
    return s[:i] + _OCR[s[i]] + s[i + 1:]


def _case(s: str, rng: random.Random) -> str:
    return rng.choice([s.upper(), s.lower(), s.title()])


def _ws(s: str, rng: random.Random) -> str:
    return rng.choice([" " * rng.randint(1, 2) + s, s + " " * rng.randint(1, 2),
                       s.replace(" ", "  ", 1) if " " in s else " " + s])


INJECTORS = {"typo": _typo, "ocr": _ocr, "case": _case, "whitespace": _ws}


def _categorical_text_cols(df, max_cols: int = 12) -> list[str]:
    """Text columns whose values RECUR (canonicalization is meaningful)."""
    out = []
    for c in df.columns:
        vals = [str(v).strip() for v in df[c].tolist() if str(v).strip()]
        if len(vals) < 20:
            continue
        alpha = sum(1 for v in vals if any(ch.isalpha() for ch in v)) / len(vals)
        nonnum = 0
        for v in vals:
            try:
                float(v.replace(",", ""))
            except ValueError:
                nonnum += 1
        if alpha < 0.7 or nonnum / len(vals) < 0.7:
            continue
        if len(set(vals)) / len(vals) > 0.5:       # must recur (categorical)
            continue
        out.append(c)
        if len(out) >= max_cols:
            break
    return out


def inject(clean_df, error_type: str, seed: int, rate: float = 0.07):
    """Return a dirty copy of `clean_df` with `error_type` errors injected into a
    `rate` fraction of cells in its categorical-text columns, or None if no eligible
    column. The original `clean_df` is the exact ground truth."""
    fn = INJECTORS[error_type]
    cols = _categorical_text_cols(clean_df)
    if not cols:
        return None
    rng = random.Random(seed)
    dirty = clean_df.copy()
    touched = 0
    for c in cols:
        col = dirty[c].tolist()
        for i, v in enumerate(col):
            s = str(v)
            if s.strip() and rng.random() < rate:
                nv = fn(s, rng)
                if nv != s:
                    col[i] = nv
                    touched += 1
        dirty[c] = col
    return dirty if touched else None
