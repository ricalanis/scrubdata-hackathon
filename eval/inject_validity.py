"""W4.5 inject-validity (TableEG-style) — does the injected slice LOOK like and RANK
like the real slice?

(1) Classifies every real dirty->gold cell error (hospital's 509 + all 42 paired
sources eval/paired_bench.py walks) with a deterministic taxonomy (typo/edit-dist<=2,
case-only, whitespace, encoding/mojibake, numeric, date-format, token-swap, missing,
other); (2) classifies the suite's INJECTED errors at the money-table seeds (7/17/27);
(3) reports Jensen-Shannon divergence (base 2) between injected and real type
distributions, pooled and per real source; (4) reports Kendall tau-b between system
rankings on the injected vs real F1 slices of money_table_head.json, with degenerate
policies (abstain-all / random-edit / oracle) run through the same suite as anchors.
Honesty rule: if the injector is far from real (high JSD), that IS the result — the
paper's mitigation (both slices reported separately) already stands.

    uv run python -m eval.inject_validity              # full run (~15 min CPU)
    uv run python -m eval.inject_validity --tex-only   # rebuild the snippet from JSON
Writes eval/results/inject_validity.json + eval/results/inject_validity_appendix.tex.
"""

from __future__ import annotations

import collections
import json
import math
import time
from datetime import datetime
from pathlib import Path

from .degenerate import _abstain_all, _oracle, _random_edit
from .metrics import _cell_equal
from .paired_bench import _load, pairs
from .run_real_multi import build_suite, score

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (7, 17, 27)            # money-table seeds (run_real_multi.main)
CATS = ["typo", "case", "whitespace", "encoding", "numeric", "date-format",
        "token-swap", "missing", "other"]
EXPECT = {"typo": "typo", "ocr": "typo", "case": "case", "whitespace": "whitespace"}
_MOJI = ("�", "Ã", "Â", "â€", "ï¿")
_DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%Y/%m/%d",
              "%d-%m-%Y", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%Y%m%d")


def _num(s: str):
    t = s.strip().replace(",", "").lstrip("$").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def _date(s: str):
    for f in _DATE_FMTS:
        try:
            return datetime.strptime(s.strip(), f).date()
        except ValueError:
            pass
    return None


def _lev_gt2(a: str, b: str) -> bool:
    """True iff Levenshtein(a, b) > 2 (banded DP, O(len*5))."""
    k = 2
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return True
    INF = k + 1
    prev = [min(j, INF) for j in range(lb + 1)]
    for i in range(1, la + 1):
        lo, hi = max(1, i - k), min(lb, i + k)
        cur = [INF] * (lb + 1)
        if i <= k:
            cur[0] = i
        for j in range(lo, hi + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]), INF)
        prev = cur
        if min(prev[max(0, lo - 1):hi + 1]) >= INF:
            return True
    return prev[lb] > k


def classify(d, g) -> str:
    """Deterministic error type from (dirty, gold) cell pair. Order matters:
    surface classes first, then value classes, edit-distance last."""
    ds, gs = str(d), str(g)
    if not ds.strip() or not gs.strip():
        return "missing"
    if "".join(ds.split()) == "".join(gs.split()):
        return "whitespace"
    if "".join(ds.split()).casefold() == "".join(gs.split()).casefold():
        return "case"
    if any(m in ds for m in _MOJI) != any(m in gs for m in _MOJI):
        return "encoding"
    if _num(ds) is not None and _num(gs) is not None:
        return "numeric"
    dd, gd = _date(ds), _date(gs)
    if dd is not None and dd == gd:
        return "date-format"
    dt, gt = sorted(ds.casefold().split()), sorted(gs.casefold().split())
    if dt == gt and len(dt) > 1:
        return "token-swap"
    if not _lev_gt2(ds.strip(), gs.strip()):
        return "typo"
    return "other"


def _classify_pair(dirty, clean) -> collections.Counter:
    n = min(len(dirty), len(clean))
    c = collections.Counter()
    for j in range(dirty.shape[1]):
        for i in range(n):
            dv, cv = dirty.iat[i, j], clean.iat[i, j]
            if not _cell_equal(dv, cv):
                c[classify(dv, cv)] += 1
    return c


def _jsd(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence, base 2 (0 = identical, 1 = disjoint)."""
    sp, sq = sum(p.values()), sum(q.values())
    out = 0.0
    for k in set(p) | set(q):
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            out += 0.5 * a * math.log2(a / m)
        if b:
            out += 0.5 * b * math.log2(b / m)
    return out


def _tau_b(xs, ys) -> float:
    """Kendall tau-b (tie-corrected); n is small, O(n^2) is fine."""
    n0 = nc = nd = tx = ty = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            n0 += 1
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            tx += a == 0
            ty += b == 0
            nc += a * b > 0
            nd += a * b < 0
    den = ((n0 - tx) * (n0 - ty)) ** 0.5
    return (nc - nd) / den if den else 0.0


def _dist(counter) -> dict:
    tot = sum(counter.values())
    return {k: round(counter.get(k, 0) / tot, 4) for k in CATS} if tot else {}


def _suite_slices(cleaner) -> tuple[float, float]:
    """(real-slice mean F1, injected-slice mean F1 over SEEDS) for a degenerate
    cleaner(dirty, clean) -> out, mirroring run_real_multi's by-source means."""
    real = []
    for spec in build_suite(seed=SEEDS[0]):
        if spec["source"] != "real":
            continue
        dirty, clean = spec["load"]()
        real.append(score(dirty, clean, cleaner(dirty, clean))["f1"])
    inj = []
    for s in SEEDS:
        fs = []
        for spec in build_suite(seed=s):
            if spec["source"] != "injected":
                continue
            loaded = spec["load"]()
            if loaded is None:
                continue
            dirty, clean = loaded
            fs.append(score(dirty, clean, cleaner(dirty, clean))["f1"])
        inj.append(sum(fs) / len(fs))
    return sum(real) / len(real), sum(inj) / len(inj)


def _write_tex(out: dict, res: Path) -> None:
    rd, jd = out["real"]["pooled_dist"], out["injected"]["pooled_dist"]
    j, rk = out["jsd"], out["ranking"]
    L = [r"% Auto-generated by eval/inject_validity.py — do not edit by hand.",
         r"\subsection{Validity of the Injected Slice}\label{app:inject-validity}",
         r"Following the TableEG-style audit, we classify every error cell (dirty vs.\ gold)",
         r"with a deterministic taxonomy and compare the suite's injected errors (money-table",
         r"seeds " + "/".join(map(str, out["seeds"])) + r", $n=" +
         f"{out['injected']['n']:,}".replace(",", r"{,}") + r"$) against the $" +
         f"{out['real']['n']:,}".replace(",", r"{,}") +
         r"$ real errors across the 42 paired sources (hospital's " +
         f"{out['real']['hospital_n']}" + r" included).",
         r"\begin{table}[t]\centering\small",
         r"\caption{Error-type distributions, real vs.\ injected (pooled).}",
         r"\label{tab:inject-validity}",
         r"\begin{tabular}{lrr}\toprule",
         r"error type & real & injected \\ \midrule"]
    for c in CATS:
        L.append(f"{c} & {rd.get(c, 0):.3f} & {jd.get(c, 0):.3f} " + r"\\")
    L += [r"\bottomrule\end{tabular}\end{table}",
          r"The injector covers only the recoverable surface classes it targets by design",
          r"(typo/case/whitespace; injector--taxonomy agreement " +
          f"{out['injected']['injector_taxonomy_agreement']:.3f}" + r"), whereas real errors",
          r"are dominated by substitutions beyond edit distance~2 (other, " +
          f"{rd['other']:.3f}" + r") and short typos (" + f"{rd['typo']:.3f}" +
          r"), with numeric (" + f"{rd['numeric']:.3f}" + r"), missing-value (" +
          f"{rd['missing']:.3f}" + r"), and encoding classes the injector never produces.",
          r"Pooled Jensen--Shannon divergence is " + f"{j['pooled']:.3f}" +
          r"~bits (per-source median " + f"{j['median']:.3f}" + r", range " +
          f"{j['min']:.3f}" + r"--" + f"{j['max']:.3f}" + r"; hospital " +
          f"{j['hospital_vs_injected']:.3f}" + r"): the two slices are \emph{not}",
          r"interchangeable, which is why the paper reports them separately and localizes",
          r"the grounding claim in the real slice. Ranking preservation is partial: Kendall",
          r"$\tau_b$ between system rankings on the injected vs.\ real F1 slices is $" +
          f"{rk['kendall_tau_b_money_table']:.2f}" + r"$ over the four cross-system rows and $" +
          f"{rk['kendall_tau_b_with_anchors']:.2f}" + r"$ with the degenerate anchors",
          r"(abstain-all, random-edit, oracle) included. The injected slice preserves the",
          r"floor/ceiling ordering but ranks OpenRefine fingerprint above both our system",
          r"and OpenRefine kNN, the reverse of the real slice --- frequency clustering looks",
          r"strong exactly where the canonical form is present and dominant by construction.",
          r"Injected-only evaluation would therefore overstate frequency-clustering",
          r"baselines."]
    (res / "inject_validity_appendix.tex").write_text("\n".join(L) + "\n")


def main() -> None:
    t0 = time.perf_counter()
    # (1) real errors: all 42 paired sources (hospital included -> its 509)
    real_per: dict[str, collections.Counter] = {}
    for p in pairs():
        try:
            dirty, clean = _load(p)
        except Exception as e:  # noqa: BLE001
            print(f"  {p.name}: LOAD FAILED {type(e).__name__}")
            continue
        real_per[p.name] = _classify_pair(dirty, clean)
        print(f"  real {p.name:<46} n={sum(real_per[p.name].values())}", flush=True)
    real_pool = sum(real_per.values(), collections.Counter())
    t_real = time.perf_counter() - t0

    # (2) injected errors at the money-table seeds, via the SAME suite generator
    inj_pool = collections.Counter()
    inj_per_injector: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    inj_per_seed = {}
    for s in SEEDS:
        cs = collections.Counter()
        for spec in build_suite(seed=s):
            if spec["source"] != "injected":
                continue
            loaded = spec["load"]()
            if loaded is None:
                continue
            dirty, clean = loaded
            c = _classify_pair(dirty, clean)
            cs += c
            inj_per_injector[spec["name"].split(":")[1]] += c
        inj_per_seed[s] = sum(cs.values())
        inj_pool += cs
        print(f"  injected seed={s} n={inj_per_seed[s]}", flush=True)
    agree = sum(inj_per_injector[et][want] for et, want in EXPECT.items())
    t_inj = time.perf_counter() - t0 - t_real

    # (3) distribution similarity
    jsd_per_source = {k: round(_jsd(real_per[k], inj_pool), 4)
                      for k in sorted(real_per) if real_per[k]}
    jsd_vals = sorted(jsd_per_source.values())
    # (4) ranking preservation: money-table systems + degenerate anchors
    money = json.load(open(ROOT / "eval" / "results" / "money_table_head.json"))
    systems = [{"system": r["system"], "real_f1": r["real_f1"], "inj_f1": r["inj_f1"],
                "anchor": False} for r in money]
    for name, fn in [("abstain-all", _abstain_all), ("random-edit", _random_edit),
                     ("oracle", _oracle)]:
        rf, jf = _suite_slices(fn)
        systems.append({"system": name, "real_f1": rf, "inj_f1": jf, "anchor": True})
        print(f"  anchor {name:<12} real={rf:.3f} inj={jf:.3f}", flush=True)
    tau_money = _tau_b([s["real_f1"] for s in systems if not s["anchor"]],
                       [s["inj_f1"] for s in systems if not s["anchor"]])
    tau_all = _tau_b([s["real_f1"] for s in systems], [s["inj_f1"] for s in systems])

    out = {
        "taxonomy": CATS, "seeds": list(SEEDS),
        "real": {"n": sum(real_pool.values()), "n_sources": len(real_per),
                 "hospital_n": sum(real_per.get("hospital", {}).values()),
                 "pooled_counts": dict(real_pool), "pooled_dist": _dist(real_pool),
                 "per_source": {k: {"n": sum(v.values()), "dist": _dist(v)}
                                for k, v in sorted(real_per.items())}},
        "injected": {"n": sum(inj_pool.values()), "per_seed_n": inj_per_seed,
                     "pooled_counts": dict(inj_pool), "pooled_dist": _dist(inj_pool),
                     "per_injector_dist": {k: _dist(v)
                                           for k, v in sorted(inj_per_injector.items())},
                     "injector_taxonomy_agreement": round(agree / sum(inj_pool.values()), 4)},
        "jsd": {"pooled": round(_jsd(real_pool, inj_pool), 4),
                "hospital_vs_injected": round(_jsd(real_per["hospital"], inj_pool), 4),
                "per_real_source_vs_injected": jsd_per_source,
                "min": jsd_vals[0], "median": jsd_vals[len(jsd_vals) // 2],
                "max": jsd_vals[-1]},
        "ranking": {"systems": systems,
                    "kendall_tau_b_money_table": round(tau_money, 4),
                    "kendall_tau_b_with_anchors": round(tau_all, 4)},
        "sec": {"real_classify": round(t_real, 1), "injected_classify": round(t_inj, 1),
                "total": round(time.perf_counter() - t0, 1)},
    }
    res = ROOT / "eval" / "results"
    json.dump(out, open(res / "inject_validity.json", "w"), indent=1)
    _write_tex(out, res)
    print(f"JSD pooled={out['jsd']['pooled']} tau(money)={tau_money:.3f} "
          f"tau(+anchors)={tau_all:.3f} -> {res / 'inject_validity.json'} "
          f"+ inject_validity_appendix.tex ({out['sec']['total']}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-only", action="store_true",
                    help="rebuild the LaTeX snippet from the existing JSON")
    if ap.parse_args().tex_only:
        res = ROOT / "eval" / "results"
        _write_tex(json.load(open(res / "inject_validity.json")), res)
        print(f"-> {res / 'inject_validity_appendix.tex'}")
    else:
        main()
