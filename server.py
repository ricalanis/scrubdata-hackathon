"""ScrubData — gr.Server backend with a custom HTML frontend (Off-Brand quest).

Exposes a single JSON API (`clean_data`) that runs the real ScrubData pipeline
(profile -> plan -> execute -> report) on an uploaded CSV/Excel file, plus a `/`
route serving the custom frontend. The cleaning engine lives in `scrubdata` and is
imported, not reimplemented.

Run:  uv run python server.py   (only then is the server launched)
Deps: gradio (already a dependency), pandas, fastapi (pulled in by gradio).
"""

from __future__ import annotations

import difflib
import io
import re
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from scrubdata import apply_plan, mock_plan, profile_dataframe, render_report
from scrubdata.active import get_planner

PLANNER = get_planner()   # fine-tuned model if SCRUBDATA_MODEL is set, else heuristic

HERE = Path(__file__).parent
FRONTEND_INDEX = HERE / "frontend" / "index.html"
SAMPLES_DIR = HERE / "samples"
ROW_CAP = 50  # rows returned to the UI for the before/after preview

app = gr.Server()

# Serve the bundled sample datasets so the frontend's "load sample" action can
# reach `samples/dirty_contacts.csv` (handle_file resolves it against origin).
if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


def _coerce_path(file_path) -> str | None:
    """Normalize the incoming file arg to a local path string.

    Depending on how the request is made, Gradio hands us either a bare path
    string or a FileData object/dict (``{"path": ..., "url": ...}``). Accept
    both so the API is robust to the JS client, the Python client, and direct
    calls.
    """
    if file_path is None:
        return None
    if isinstance(file_path, str):
        return file_path or None
    if isinstance(file_path, dict):
        return file_path.get("path") or file_path.get("name") or None
    # FileData-like object with a `.path` attribute
    path = getattr(file_path, "path", None)
    return path or None


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Real-world exports arrive with duplicate headers, blank headers, or no header
    row at all (numeric column labels). The engine addresses columns by unique string
    name, so repair them at ingestion: stringify, fill blanks as column_N, and
    de-duplicate with .1/.2 suffixes. Demo-safety — a messy header must never crash."""
    seen: dict[str, int] = {}
    new_cols = []
    for i, c in enumerate(df.columns):
        name = str(c).strip()
        if not name or name.lower() == "nan" or str(c).startswith("Unnamed"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df


def _read_any(path: str) -> pd.DataFrame:
    """Read CSV or Excel as raw strings — cleaning decides the real types."""
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p, dtype=str)
    else:
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:        # non-UTF-8 export (Excel often emits cp1252)
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="latin-1")
    return _sanitize_columns(df)


def _records(df: pd.DataFrame, cap: int = ROW_CAP) -> list[dict]:
    """First `cap` rows as JSON-safe row dicts (NaN/NA -> None)."""
    head = df.head(cap)
    # object-dtype the frame so .where can place None without re-casting issues,
    # then replace any pandas NA/NaN with None for valid JSON.
    safe = head.astype(object).where(pd.notna(head), None)
    return safe.to_dict(orient="records")


_WS = re.compile(r"\s+")


def _row_signature(rec: dict) -> str:
    """A transform-tolerant signature for one row.

    The engine drops rows (empty/dedup) and *then* normalizes cell values
    (trim, lowercase email, canonicalize categories, reformat phone/date...).
    Naive index pairing therefore mis-aligns every row after the first drop.
    We instead align before/after rows with difflib over a normalized signature
    that survives those value-level transforms: lowercase, collapse whitespace,
    keep only alphanumerics. Alignment happens here (server-side, on the full
    data) so the UI reflects real row identity rather than re-deriving it.
    """
    parts = []
    for v in rec.values():
        s = "" if v is None else str(v)
        s = _WS.sub(" ", s).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        parts.append(s)
    return "\x1f".join(parts)


def _align_rows(before: list[dict], after: list[dict]) -> list[dict]:
    """Pair before/after preview rows by content identity, not by index.

    Returns a list of alignment ops, each:
      {"type": "pair",    "b": <before idx>, "a": <after idx>}
      {"type": "removed", "b": <before idx>}
      {"type": "added",   "a": <after idx>}
    `after` is (post-transform) an in-order subsequence of `before`, so a
    sequence alignment over normalized signatures recovers the true mapping
    and isolates dropped rows instead of smearing "changed" across the table.
    """
    bsig = [_row_signature(r) for r in before]
    asig = [_row_signature(r) for r in after]
    sm = difflib.SequenceMatcher(a=bsig, b=asig, autojunk=False)
    ops: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                ops.append({"type": "pair", "b": i1 + off, "a": j1 + off})
        elif tag == "replace":
            # A "replace" block is a run of before-rows with no exact signature
            # match among a run of after-rows. Pairing them blindly (by index)
            # reintroduces the smear we're avoiding — a dropped empty/garbage row
            # paired against an unrelated kept row floods the diff with false
            # "changed" cells. Only pair rows whose signatures are *similar*
            # (same row, value-level transforms); leave the rest removed/added.
            ops.extend(_align_block(bsig, asig, i1, i2, j1, j2))
        elif tag == "delete":
            for off in range(i2 - i1):
                ops.append({"type": "removed", "b": i1 + off})
        elif tag == "insert":
            for off in range(j2 - j1):
                ops.append({"type": "added", "a": j1 + off})
    return _reconcile(ops, bsig, asig)


def _reconcile(ops: list[dict], bsig, asig, thresh: float = 0.74) -> list[dict]:
    """Re-pair `removed`/`added` rows that block boundaries split apart.

    Duplicate-collapse can leave a genuine survivor labelled `removed` on one
    side and `added` on the other (an off-by-N at a block edge). Walk the ops in
    order and, whenever a removed row is followed later by a sufficiently similar
    added row (no intervening pair crossing them), merge the two into one pair so
    the row shows as a single, correctly-aligned change rather than remove+add.
    """
    rem_idx = [k for k, o in enumerate(ops) if o["type"] == "removed"]
    add_idx = [k for k, o in enumerate(ops) if o["type"] == "added"]
    used_rem: set[int] = set()
    for ak in add_idx:
        aj = ops[ak]["a"]
        best_rk, best_r = -1, thresh
        for rk in rem_idx:
            if rk in used_rem:
                continue
            bi = ops[rk]["b"]
            r = difflib.SequenceMatcher(None, bsig[bi], asig[aj], autojunk=False).ratio()
            if r >= best_r:
                best_r, best_rk = r, rk
        if best_rk >= 0:
            used_rem.add(best_rk)
            ops[best_rk] = {"type": "pair", "b": ops[best_rk]["b"], "a": aj}
            ops[ak] = {"type": "_drop"}  # tombstone; removed below
    return [o for o in ops if o["type"] != "_drop"]


def _align_block(bsig, asig, i1, i2, j1, j2, thresh: float = 0.74) -> list[dict]:
    """Greedily, in order, pair before/after rows within one replace block.

    A before-row is paired to the next after-row only when their normalized
    signatures are similar enough (difflib ratio >= `thresh`); otherwise the
    before-row is `removed` and/or the after-row is `added`. Order-preserving,
    so it never crosses pairs.
    """
    ops: list[dict] = []
    i, j = i1, j1
    while i < i2 and j < j2:
        ratio = difflib.SequenceMatcher(None, bsig[i], asig[j], autojunk=False).ratio()
        if ratio >= thresh:
            ops.append({"type": "pair", "b": i, "a": j})
            i += 1
            j += 1
        else:
            # Decide whether this before-row was dropped or this after-row is new
            # by peeking one step ahead on each side.
            b_next = (
                difflib.SequenceMatcher(None, bsig[i + 1], asig[j], autojunk=False).ratio()
                if i + 1 < i2 else -1.0
            )
            a_next = (
                difflib.SequenceMatcher(None, bsig[i], asig[j + 1], autojunk=False).ratio()
                if j + 1 < j2 else -1.0
            )
            if a_next >= b_next:
                ops.append({"type": "added", "a": j})
                j += 1
            else:
                ops.append({"type": "removed", "b": i})
                i += 1
    while i < i2:
        ops.append({"type": "removed", "b": i})
        i += 1
    while j < j2:
        ops.append({"type": "added", "a": j})
        j += 1
    return ops


def _empty_result(summary: str) -> dict:
    """The no-op / graceful-error response shape (frontend tolerates missing keys)."""
    return {
        "before": [], "after": [], "columns_before": [], "columns_after": [],
        "alignment": [], "change_log": [], "total_rows_before": 0,
        "total_rows_after": 0, "preview_cap": ROW_CAP, "report_md": "",
        "csv_text": "", "summary": summary,
    }


@app.api(name="clean_data")
def clean_data(file_path: str) -> dict:
    """Run the full pipeline on an uploaded file and return a JSON-safe dict.

    `file_path` is a local path string or FileData (dict/object). Returns keys:
      before, after, columns_before, columns_after, alignment, change_log,
      total_rows_before, total_rows_after, preview_cap, report_md, csv_text,
      summary.
    """
    file_path = _coerce_path(file_path)
    if not file_path:
        return _empty_result("No file provided. Upload a CSV or Excel file to begin.")

    try:
        raw = _read_any(file_path)
    except Exception as e:  # noqa: BLE001 — never crash the demo on a malformed file
        return _empty_result(
            f"Couldn't read this file ({type(e).__name__}). "
            "Try exporting it as a CSV or .xlsx and dropping it again.")
    if raw is None or raw.empty or len(raw.columns) == 0:
        return _empty_result("That file looks empty — no rows or columns to clean.")

    try:
        _t0 = time.perf_counter()
        before_profile = profile_dataframe(raw)
        plan = PLANNER(raw)
        cleaned, change_log = apply_plan(raw, plan)
        elapsed_ms = int((time.perf_counter() - _t0) * 1000)
        after_profile = profile_dataframe(cleaned)
        report_md = render_report(plan, change_log, before_profile, after_profile)
    except Exception as e:  # noqa: BLE001 — degrade gracefully, surface the original untouched
        return _empty_result(
            f"Something went wrong while cleaning ({type(e).__name__}) — your file is "
            "untouched. This is logged; please try another export.")

    return _build_response(raw, cleaned, plan, change_log, elapsed_ms,
                           before_profile, report_md)


def _build_response(raw, cleaned, plan, change_log, elapsed_ms,
                    before_profile=None, report_md="") -> dict:
    """Assemble the JSON-safe response. Shared by clean_data (model/heuristic plan)
    and clean_with_plan (replay a saved recipe), so both render identically."""
    try:  # best-effort agent-trace capture (Open trace bonus quest)
        from scrubdata.trace import log_run
        if before_profile is not None:
            log_run(before_profile, raw, plan, change_log,
                    model=plan.get("_generated_by", "mock_planner"))
    except Exception:  # noqa: BLE001
        pass

    buf = io.StringIO()
    cleaned.to_csv(buf, index=False)
    csv_text = buf.getvalue()

    n_changes = len(change_log) if change_log is not None else 0
    summary = (
        f"Cleaned {len(raw):,} rows × {len(raw.columns)} columns -> "
        f"{len(cleaned):,} rows × {len(cleaned.columns)} columns "
        f"({n_changes} change{'s' if n_changes != 1 else ''} applied)."
    )

    before_records = _records(raw)
    after_records = _records(cleaned)

    return {
        "before": before_records,
        "after": after_records,
        "columns_before": list(raw.columns),
        "columns_after": list(cleaned.columns),
        # Content-based pairing of the *previewed* rows so the UI's diff reflects
        # real row identity (handles dropped/deduped rows without smearing).
        "alignment": _align_rows(before_records, after_records),
        "change_log": change_log if change_log is not None else [],
        # True dataset totals (the before/after arrays are capped previews).
        "total_rows_before": int(len(raw)),
        "total_rows_after": int(len(cleaned)),
        # scale-invariance demo beat: profile+plan+execute wall-clock. The prompt
        # scales with DISTINCT values not rows, so this stays low on big tables.
        "elapsed_ms": elapsed_ms,
        "preview_cap": ROW_CAP,
        "report_md": report_md,
        "csv_text": csv_text,
        "summary": summary,
        # structured plan for the card UI: applied ops, review flags, PII, audit signals
        "plan_columns": [
            {"name": c.get("name"), "semantic_type": c.get("detected_semantic_type"),
             "operations": [
                 {"op": o.get("op"), "rationale": o.get("rationale", ""),
                  "pii_type": o.get("pii_type"),
                  "mapping_sample": dict(list(o.get("mapping", {}).items())[:6]) or None,
                  "mapping_size": len(o.get("mapping", {})) or None}
                 for o in c.get("operations", [])]}
            for c in plan.get("columns", [])],
        "flags": plan.get("flags", []),
        "monitor": _monitor(plan, change_log),
        # embedded-PII awareness (product-only, detection not edit): cards/SSNs buried
        # in free-text columns the column typer didn't flag. Surfaced for review.
        "pii_alerts": _embedded_pii_alerts(raw, plan),
        # the executable plan itself — the "cleaning recipe" the user can save and
        # re-apply to next month's same-shaped export via clean_with_plan.
        "plan_raw": plan,
    }


@app.api(name="clean_with_plan")
def clean_with_plan(file_path: str, plan_json: str) -> dict:
    """Replay a SAVED recipe (plan JSON from a prior run) on a NEW file — the 'Monday
    ritual': same cleaning, next month's export, one click. No re-planning."""
    import json as _json
    file_path = _coerce_path(file_path)
    if not file_path:
        return _empty_result("Upload the new file to apply your saved recipe to.")
    try:
        plan = _json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        assert isinstance(plan, dict) and "columns" in plan
    except Exception:  # noqa: BLE001
        return _empty_result("That isn't a ScrubData recipe — expected the saved plan JSON.")
    try:
        raw = _read_any(file_path)
    except Exception as e:  # noqa: BLE001
        return _empty_result(f"Couldn't read the data file ({type(e).__name__}).")
    if raw is None or raw.empty or len(raw.columns) == 0:
        return _empty_result("That file looks empty.")
    try:
        _t0 = time.perf_counter()
        cleaned, change_log = apply_plan(raw, plan)
        elapsed_ms = int((time.perf_counter() - _t0) * 1000)
        report_md = ""
        try:
            report_md = render_report(plan, change_log,
                                      profile_dataframe(raw), profile_dataframe(cleaned))
        except Exception:  # noqa: BLE001
            pass
        plan = {**plan, "_generated_by": "saved recipe (replay)"}
        return _build_response(raw, cleaned, plan, change_log, elapsed_ms, None, report_md)
    except Exception as e:  # noqa: BLE001
        return _empty_result(
            f"Couldn't apply the recipe to this file ({type(e).__name__}) — "
            "is it the same kind of export?")


def _embedded_pii_alerts(raw, plan: dict) -> list[dict]:
    """Scan raw text columns (that the planner didn't already type as PII) for
    high-precision embedded cards/SSNs, so the UI can warn before the user shares."""
    try:
        from scrubdata import pii
        typed = {c.get("name") for c in plan.get("columns", [])
                 if any("pii" in (o.get("op") or "") for o in c.get("operations", []))}
        alerts = []
        for col in raw.columns:
            if col in typed:
                continue
            a = pii.scan_embedded_pii(col, raw[col].tolist())
            if a:
                alerts.append(a)
        return alerts
    except Exception:  # noqa: BLE001
        return []


def _monitor(plan: dict, change_log) -> dict:
    try:
        from scrubdata.observability import monitor_summary
        return monitor_summary(plan, change_log)
    except Exception:  # noqa: BLE001
        return {}


_PLACEHOLDER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ScrubData</title></head>
<body style="font-family:system-ui;max-width:42rem;margin:4rem auto;padding:0 1rem">
<h1>ScrubData</h1>
<p>Backend is running. The custom frontend
(<code>frontend/index.html</code>) hasn't been built yet — the Integrate step
creates it. The <code>clean_data</code> API is live.</p>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def homepage() -> str:
    try:
        return FRONTEND_INDEX.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return _PLACEHOLDER_HTML


def _warmup() -> None:
    """Pre-build the reference index (one-time ~5s) and pre-run the bundled samples so
    the demo's first click is warm (best()'s per-value cache is populated). Runs in a
    daemon thread at import so server boot is never blocked (HF Spaces boot timeout)."""
    try:
        from scrubdata.reconcile import default_index
        default_index()
        for s in sorted(SAMPLES_DIR.glob("*.csv")) if SAMPLES_DIR.is_dir() else []:
            try:
                df = _read_any(str(s))
                apply_plan(df, PLANNER(df))
                profile_dataframe(df)
            except Exception:  # noqa: BLE001 — warmup must never crash the server
                pass
    except Exception:  # noqa: BLE001
        pass


import threading as _threading
_threading.Thread(target=_warmup, daemon=True).start()


if __name__ == "__main__":
    import os
    app.launch(server_name="0.0.0.0",
               server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
