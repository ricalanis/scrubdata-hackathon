"""ScrubData — hands-off data cleaning (Gradio app).

Runnable MOCK demo on gr.Blocks: upload → profile → plan → clean → diff +
report → download. The planner is a heuristic stand-in for the fine-tuned ≤4B
model; the rest of the pipeline is real. Final version will port this flow to
gr.Server + a custom HTML frontend for the Off-Brand bonus quest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from scrubdata import apply_plan, mock_plan, profile_dataframe, render_report
from scrubdata.active import get_planner
from scrubdata.trace import log_run

PLANNER = get_planner()   # fine-tuned model if SCRUBDATA_MODEL is set, else heuristic

SAMPLE = Path(__file__).parent / "samples" / "dirty_contacts.csv"


def _read_any(path: str) -> pd.DataFrame:
    """Read CSV or Excel as raw strings (cleaning decides the real types)."""
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p, dtype=str)
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def clean(file_path: str):
    if not file_path:
        return (gr.update(), gr.update(), "Upload a CSV or Excel file to begin.", None)

    raw = _read_any(file_path)
    before = profile_dataframe(raw)
    plan = PLANNER(raw)
    cleaned, log = apply_plan(raw, plan)
    after = profile_dataframe(cleaned)
    report = render_report(plan, log, before, after)

    out = Path(tempfile.gettempdir()) / "scrubbed.csv"
    cleaned.to_csv(out, index=False)

    try:  # best-effort agent-trace capture (Open trace bonus quest)
        log_run(before, raw, plan, log, model=plan.get("_generated_by", "mock_planner"))
    except Exception:
        pass

    return raw, cleaned, report, str(out)


def load_sample():
    return str(SAMPLE)


with gr.Blocks(title="ScrubData") as demo:
    gr.Markdown(
        "# 🧽 ScrubData\n"
        "**Upload your dirty spreadsheet. Get clean data back. No config.**\n\n"
        "_Mock demo — heuristic planner standing in for the fine-tuned model._"
    )

    with gr.Row():
        file_in = gr.File(label="Upload CSV / Excel", file_types=[".csv", ".xlsx", ".xls"],
                          type="filepath")
        with gr.Column():
            run_btn = gr.Button("🧽 Clean it", variant="primary")
            sample_btn = gr.Button("Use the messy sample")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Before")
            before_df = gr.Dataframe(label="Original", interactive=False, wrap=True)
        with gr.Column():
            gr.Markdown("### After")
            after_df = gr.Dataframe(label="Cleaned", interactive=False, wrap=True)

    report_md = gr.Markdown()
    download = gr.File(label="Download cleaned file")

    run_btn.click(clean, inputs=file_in, outputs=[before_df, after_df, report_md, download])
    sample_btn.click(load_sample, outputs=file_in)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
