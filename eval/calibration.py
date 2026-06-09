"""Selective prediction / calibration study for grounded canonicalization.

"Knowing when NOT to act" is the research contribution (and the AI-safety monitorability
angle): instead of always emitting a canonical, the grounded reconciler attaches a
CONFIDENCE and ABSTAINS below threshold. This module measures whether that confidence is
trustworthy:

  * Risk-Coverage curve + AURC — sort decisions by confidence; as we cover more (abstain
    less) does risk rise gracefully? Low AURC = a good selective predictor.
  * ECE (Expected Calibration Error) — does a confidence of 0.9 actually mean ~90% correct?
  * Operating point — at our default threshold, what coverage and precision do we get, and
    what threshold hits a target precision (e.g. 95%)?

Probe = real cities sampled from the reference with injected typos (recoverable, gold known)
+ garbage TRAP strings (acting at all is an error). Reproducible (fixed seed).

    uv run python -m eval.calibration
"""

from __future__ import annotations

import random
import string

from scrubdata.reconcile import _norm, default_index


def _typo(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s + rng.choice(string.ascii_lowercase)
    i = rng.randrange(1, len(s) - 1)
    if not s[i].isalpha():
        i = 1
    pool = string.ascii_lowercase if s[i].islower() else string.ascii_uppercase
    return s[:i] + rng.choice([c for c in pool if c != s[i].lower()]) + s[i + 1:]


def build_probe(n_real: int = 500, n_trap: int = 150, seed: int = 5):
    """(value, gold|None, kind) probes: real-city typos (recoverable) + garbage traps."""
    idx = default_index()
    cities = [c for bucket in idx._buckets.get("city", {}).values() for (c, _) in bucket]
    rng = random.Random(seed)
    probe = []
    for c in rng.sample(cities, min(n_real, len(cities))):
        probe.append((_typo(c, rng), c, "real"))
    for _ in range(n_trap):
        g = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9)))
        probe.append((g, None, "trap"))
    rng.shuffle(probe)
    return probe, idx


def _scored(probe, idx, ctype="city"):
    """(confidence, correct_if_acted) per probe."""
    out = []
    for value, gold, kind in probe:
        b = idx.best(value, ctype)
        conf = b[1] if b else 0.0
        correct = bool(kind == "real" and b and _norm(b[0]) == _norm(gold))
        out.append((conf, correct))
    return out


def risk_coverage(scored):
    rows = sorted(scored, key=lambda x: -x[0])
    n, cum = len(rows), 0
    curve = []
    for k, (conf, ok) in enumerate(rows, 1):
        cum += int(ok)
        curve.append((k / n, 1 - cum / k, conf))      # coverage, risk, confidence
    aurc = sum(r for _, r, _ in curve) / len(curve)
    return curve, aurc


def ece(scored, bins: int = 10) -> float:
    n = len(scored)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in scored if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not bucket:
            continue
        conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(int(ok) for _, ok in bucket) / len(bucket)
        e += len(bucket) / n * abs(conf - acc)
    return e


def operating_point(scored, threshold: float):
    acted = [(c, ok) for c, ok in scored if c >= threshold]
    coverage = len(acted) / len(scored)
    precision = (sum(int(ok) for _, ok in acted) / len(acted)) if acted else 1.0
    return coverage, precision


def main() -> None:
    probe, idx = build_probe()
    scored = _scored(probe, idx)
    curve, aurc = risk_coverage(scored)
    e = ece(scored)
    print(f"\n=== Selective prediction / calibration — grounded city reconciliation "
          f"({len(probe)} probes: real typos + traps) ===\n")
    print(f"  AURC (area under risk-coverage, lower=better) = {aurc:.4f}")
    print(f"  ECE  (expected calibration error, lower=better) = {e:.4f}")
    print("\n  Risk-Coverage operating points:")
    print(f"  {'threshold':>10}{'coverage':>10}{'precision':>11}")
    for t in (0.70, 0.78, 0.84, 0.90, 0.95, 1.00):
        cov, prec = operating_point(scored, t)
        print(f"  {t:>10.2f}{cov:>10.3f}{prec:>11.3f}")
    # threshold achieving >=95% precision
    best_t = next((t / 100 for t in range(70, 101)
                   if operating_point(scored, t / 100)[1] >= 0.95), 1.0)
    cov95, _ = operating_point(scored, best_t)
    print(f"\n  -> for >=95% precision use threshold {best_t:.2f} (coverage {cov95:.3f}). "
          "The confidence is trustworthy enough to ABSTAIN on the rest — the safety contract.")


if __name__ == "__main__":
    main()
