"""W1.b — self-consistency + verifier re-ranking on the WS1 hospital gate.

Samples N raw plans at temperature from the fine-tuned LOCAL planner through the exact
WS1 capture composition (make_batched_planner batch_size=4, no grounded wrapper, no
union — matches eval/results/v6_hospital_raw_plan.json and the Modal capture; see
eval/capture_plan_local.py), majority-votes mappings at (column, raw->canon) cell-edit
level (keep entries in >= ceil(N/2) samples; vote share recorded), then runs the voted
plan through the SHIPPED selective-prediction pipeline — verify_plan(tau) + union with
the grounded heuristic — and scores against hospital's 509 real errors with the
eval.precision_curve machinery. Also captures one greedy (temperature 0) plan as the
reproduction anchor vs the shipped 0.905 @ 0.413. Measurement, not a ship decision.

Decoding is format=json (grammar-constrained): without it the Q8 GGUF's first token
degenerates into <tool_call> loops — the Modal bf16 captures suppressed the same two
tokens (suppress_tokens=[151657, 151658]); this is the local equivalent.

    ollama create scrubdata-ft -f notebooks/Modelfile
    uv run python -m eval.sc_rerank --model scrubdata-ft --n 8 \
        --out eval/results/sc_rerank.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter

from scrubdata.executor import apply_plan
from scrubdata.model_planner import _extract_json, make_batched_planner
from scrubdata.planner import mock_plan
from scrubdata.profiler import profile_dataframe
from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
from scrubdata.verifier import union_plans, verify_plan

from .precision_curve import TAUS, _repairs_only
from .run_real import _ensure_data, _load
from .run_real_multi import score as _cn_score

SHIPPED = {"precision": 0.905, "coverage": 0.413, "tau": 0.5}   # WS1 gate (aa48108)

NUM_PREDICT = 4000   # batch 3 needs 2122 tokens; 2000 truncated 2/5 hospital batches


def _salvage_json(text: str) -> dict | None:
    """Repair a generation truncated mid-JSON (done_reason=length): cut at the last
    structurally complete value and close the open brackets. Q8-local failure mode:
    greedy repetition loops inside a mapping never emit the closing brace; the entries
    before the loop are valid and (being duplicates) dedupe in the dict."""
    i = text.find("{")
    if i == -1:
        return None
    stack, in_str, esc = [], False, False
    cut = None                                  # (pos, closers) at last safe point
    for j, ch in enumerate(text[i:], start=i):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                cut = (j, "".join(reversed(stack)))   # after a complete string value/key
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            cut = (j, "".join(reversed(stack)))
    if not stack or cut is None:
        return None
    frag = text[i:cut[0] + 1]
    # a cut after a KEY (`"key"` with no value yet) is invalid — drop the dangling key
    for cand in (frag + cut[1],
                 frag.rsplit(",", 1)[0] + cut[1] if "," in frag else None):
        if cand is None:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def make_sampling_planner(model: str, temperature: float, seed: int,
                          host: str = "http://localhost:11434", timeout: int = 600):
    """make_local_ollama_planner with temperature + seed exposed, format=json constrained
    (blocks the degenerate <tool_call> first token, like the Modal suppress_tokens)."""
    import urllib.request

    def planner(dirty_df, *_):
        profile = profile_dataframe(dirty_df)
        user = build_user_prompt(profile, dirty_df)
        payload = {
            "model": model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "options": {"temperature": temperature, "seed": seed,
                        "num_predict": NUM_PREDICT, "num_ctx": 16384},
        }
        req = urllib.request.Request(
            host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        out, last_err = None, None
        for attempt in range(3):                # ride out transient 500s / reloads
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    out = json.loads(r.read())["message"]["content"]
                break
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:120]
                time.sleep(10 * (attempt + 1))
        if out is None:
            return {"__error__": last_err}
        plan = _extract_json(out)
        if plan is None:
            plan = _salvage_json(out)
            if plan is not None:
                plan["_salvaged"] = True
        if plan is None:
            return {"__error__": "no_json", "raw": out[:200]}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan
    return planner


def capture_raw_plan(model: str, dirty, temperature: float, seed: int,
                     host: str = "http://localhost:11434") -> tuple[dict, int]:
    """The WS1 capture composition: make_batched_planner(model, 4) — no grounded wrapper,
    no fallback (failed batches contribute nothing, as in the Modal capture). Returns
    (raw plan, n failed batches)."""
    raw = make_sampling_planner(model, temperature, seed, host=host)
    failed, salvaged = [0], [0]

    def counted(df, *_):
        p = raw(df)
        if not (isinstance(p, dict) and "__error__" not in p):
            failed[0] += 1
        elif p.pop("_salvaged", False):
            salvaged[0] += 1
        return p

    plan = make_batched_planner(counted, batch_size=4)(dirty)
    # sampling can emit malformed entries (a bare string in columns/operations):
    # drop non-dict items — the executor/verifier contract is dicts only
    plan["columns"] = [c for c in plan.get("columns", []) if isinstance(c, dict)]
    for c in plan["columns"]:
        c["operations"] = [o for o in c.get("operations", []) if isinstance(o, dict)]
    plan["flags"] = [f for f in plan.get("flags", []) if isinstance(f, dict)]
    plan["_capture"] = {"failed_batches": failed[0], "salvaged_batches": salvaged[0]}
    return plan, failed[0]


def _entries(plan: dict):
    """Yield (column, raw, canon, grounded?) for every canonicalize mapping entry."""
    for c in plan.get("columns", []):
        for o in c.get("operations", []):
            if o.get("op") != "canonicalize_categories":
                continue
            g = "reference taxonomy" in o.get("rationale", "")
            for r, cn in o.get("mapping", {}).items():
                yield (c.get("name"), str(r), str(cn), g)


def vote_plans(plans: list[dict], k: int) -> tuple[dict, dict]:
    """Majority-vote N raw plans at (column, raw->canon) cell-edit level: keep entries in
    >= k samples (grounded entries keep their rationale so the verifier passes them
    through, as in the shipped pipeline). Non-canonicalize ops and table ops are voted at
    (column, op-name) level. Returns (voted plan, vote diagnostics)."""
    n = len(plans)
    votes = Counter(e for p in plans for e in set(_entries(p)))
    kept = {e: v for e, v in votes.items() if v >= k}
    # column ops other than canonicalize, voted at op identity
    op_votes = Counter()
    op_proto: dict = {}
    for p in plans:
        seen = set()
        for c in p.get("columns", []):
            for o in c.get("operations", []):
                if o.get("op") == "canonicalize_categories":
                    continue
                key = (c.get("name"), o.get("op"))
                if key not in seen:
                    seen.add(key)
                    op_votes[key] += 1
                    op_proto.setdefault(key, (o, c))
    cols: dict = {}

    def _col(plan_col_name, proto_c):
        if plan_col_name not in cols:
            cols[plan_col_name] = {
                "name": plan_col_name,
                "detected_semantic_type": proto_c.get("detected_semantic_type", "categorical"),
                "issues": list(proto_c.get("issues", [])), "operations": []}
        return cols[plan_col_name]

    for (cname, _opn), v in sorted(op_votes.items(), key=lambda x: x[0][1] or ""):
        if v >= k:
            o, proto_c = op_proto[(cname, _opn)]
            _col(cname, proto_c)["operations"].append(json.loads(json.dumps(o)))
    proto_cols = {c.get("name"): c for p in plans for c in p.get("columns", [])}
    by_col: dict = {}
    # ascending vote order so on (column, raw) conflicts the higher-vote canon wins
    # (only reachable at k=1 / union-of-all; a majority threshold can keep one side only)
    for (cname, r, cn, g), v in sorted(kept.items(), key=lambda x: x[1]):
        by_col.setdefault((cname, g), {})[r] = cn
    for (cname, g), mapping in by_col.items():
        col = _col(cname, proto_cols.get(cname, {}))
        col["operations"].append({
            "op": "canonicalize_categories", "mapping": mapping,
            "rationale": ("Reconciled to the reference taxonomy (grounded, not "
                          "free-generated); self-consistency voted." if g else
                          f"Self-consistency majority vote over {n} samples.")})
    voted = {"dataset_summary": plans[0].get("dataset_summary", ""),
             "table_operations": json.loads(json.dumps(plans[0].get("table_operations", []))),
             "columns": list(cols.values()), "flags": []}
    diag = {"n_samples": n, "threshold": k, "entries_union": len(votes),
            "entries_kept": len(kept),
            "vote_hist": dict(Counter(votes.values())),
            "kept_vote_share": {f"{c}|{r}->{cn}": round(v / n, 3)
                                for (c, r, cn, _g), v in sorted(kept.items())}}
    return voted, diag


def gate_point(dirty, clean, base_plan: dict, tau: float = 0.5, union: bool = True) -> dict:
    """One precision-curve point: verify(tau) [-> union heuristic] -> repairs-only score."""
    plan = verify_plan(dirty, base_plan, tau=tau)
    if union:
        plan = union_plans(plan, mock_plan(dirty))
    cleaned, _ = apply_plan(dirty, _repairs_only(plan))
    m = _cn_score(dirty, clean, cleaned)
    return {"tau": tau, "precision": round(m["precision"], 4), "coverage": round(m["recall"], 4),
            "changed": m["_changed"], "fixed": m["_fixed"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scrubdata-ft")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=100, help="base sampling seed (seed+i per sample)")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="ollama host (the published v6 Q8 GGUF degenerates on ollama "
                         "0.21.2; verified working on 0.30.7)")
    ap.add_argument("--out", type=str, default="eval/results/sc_rerank.json")
    ap.add_argument("--blob-sha256-prefix", default="",
                    help="sha256 prefix of the served GGUF blob (provenance)")
    args = ap.parse_args()

    _ensure_data()
    dirty, clean = _load()

    # reproduction anchor: greedy capture through the same pipeline
    t0 = time.time()
    greedy, g_fb = capture_raw_plan(args.model, dirty, 0.0, args.seed, host=args.host)
    g_secs = time.time() - t0
    g_point = gate_point(dirty, clean, greedy)
    g_cap = greedy.get("_capture", {})
    print(f"[greedy anchor] {g_secs:.0f}s, capture={g_cap}, "
          f"union tau=0.5: {g_point['precision']:.3f} @ {g_point['coverage']:.3f} "
          f"(shipped {SHIPPED['precision']} @ {SHIPPED['coverage']})", flush=True)

    samples = []
    for i in range(args.n):
        t0 = time.time()
        plan, fb = capture_raw_plan(args.model, dirty, args.temperature,
                                    args.seed + 1 + i, host=args.host)
        if fb and not list(_entries(plan)):     # server hiccup ate the sample: one redo
            print(f"[sample {i + 1}/{args.n}] all batches failed — retrying once", flush=True)
            plan, fb = capture_raw_plan(args.model, dirty, args.temperature,
                                        args.seed + 1 + i, host=args.host)
        secs = time.time() - t0
        pt = gate_point(dirty, clean, plan)
        samples.append({"seed": args.seed + 1 + i, "secs": round(secs, 1),
                        "capture": plan.get("_capture", {}),
                        "n_entries": len(set(_entries(plan))),
                        "plan": plan, "point_tau05_union": pt})
        print(f"[sample {i + 1}/{args.n}] {secs:.0f}s, capture={samples[-1]['capture']}, "
              f"entries={samples[-1]['n_entries']}, union tau=0.5: "
              f"{pt['precision']:.3f} @ {pt['coverage']:.3f}", flush=True)
        json.dump(samples, open(args.out + ".partial", "w"))   # checkpoint

    k = math.ceil(args.n / 2)
    voted, diag = vote_plans([s["plan"] for s in samples], k)
    print(f"\n[vote] union {diag['entries_union']} entries -> kept {diag['entries_kept']} "
          f"(>= {k}/{args.n} votes); hist {diag['vote_hist']}")

    rows = [gate_point(dirty, clean, voted, tau=t) for t in TAUS]
    print(f"\n=== voted plan + verify + heuristic union (hospital, 509 real errors) ===")
    print(f"{'tau':>5}{'precision':>11}{'coverage':>10}{'changed':>9}{'fixed':>7}")
    for r in rows:
        print(f"{r['tau']:>5.2f}{r['precision']:>11.3f}{r['coverage']:>10.3f}"
              f"{r['changed']:>9}{r['fixed']:>7}")
    v_point = next(r for r in rows if r["tau"] == 0.5)
    print(f"\nvoted-union @ tau=0.5: {v_point['precision']:.3f} @ {v_point['coverage']:.3f}  "
          f"vs shipped {SHIPPED['precision']} @ {SHIPPED['coverage']}")

    # ablation (a): best single sample — max precision among points with coverage >= 0.30
    eligible = [s for s in samples if s["point_tau05_union"]["coverage"] >= 0.30]
    pool = eligible or samples           # if nothing reaches 0.30, fall back + flag it
    best = max(pool, key=lambda s: s["point_tau05_union"]["precision"])
    best_single = dict(best["point_tau05_union"],
                       seed=best["seed"], coverage_floor_met=bool(eligible))
    print(f"[ablation a] best single (seed {best['seed']}): "
          f"{best_single['precision']:.3f} @ {best_single['coverage']:.3f} "
          f"(eligible >=0.30 cov: {len(eligible)}/{len(samples)})")

    # ablation (b): union of ALL samples (k=1, conflicts -> higher-vote canon), verify+union
    union_plan, union_diag = vote_plans([s["plan"] for s in samples], 1)
    u_point = gate_point(dirty, clean, union_plan)
    print(f"[ablation b] union-of-all ({union_diag['entries_union']} entries): "
          f"{u_point['precision']:.3f} @ {u_point['coverage']:.3f}")

    out = {"model": args.model, "n_samples": args.n, "temperature": args.temperature,
           "base_seed": args.seed, "host": args.host, "threshold": k,
           "decoding": {"temperature": args.temperature, "format": "json",
                        "num_predict": NUM_PREDICT, "num_ctx": 16384},
           "model_blob_sha256_prefix": args.blob_sha256_prefix or None,
           "shipped_reference": SHIPPED,
           "greedy_anchor": {"secs": round(g_secs, 1), "failed_batches": g_fb,
                             "point_tau05_union": g_point},
           "per_sample_runtimes": [s["secs"] for s in samples],
           "per_sample": [{kk: v for kk, v in s.items() if kk != "plan"} for s in samples],
           "vote": diag, "voted_curve": rows, "voted": v_point,
           "best_single": best_single,
           "union_all": dict(u_point, entries_union=union_diag["entries_union"]),
           "voted_plan": voted}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
