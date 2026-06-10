"""WS4 baseline: Jellyfish-13B (Zhang et al., EMNLP 2024) on the Raha real slice.

Composes its two published cell-level tasks — error detection (yes/no) then data
imputation (infer the flagged cell) — into repairs scored under our churn-neutral
protocol. Prompts verbatim from the model card (built in eval/baselines_learned.py);
recommended decoding (temp 0.35, top_p 0.9, rep-penalty 1.15) via vLLM.

Caveats disclosed in the paper: hospital is in Jellyfish's training data (published ED
F1 95.6); flights/rayyan are in its eval suite; the ED+DI composition is ours.

    uv run modal run --detach scripts/modal_jellyfish.py --datasets hospital   # sanity
    uv run modal run --detach scripts/modal_jellyfish.py                       # full
"""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**",
          "data/**", "eval/results/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.11.2", "pandas", "huggingface_hub")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
)
app = modal.App("scrubdata-jellyfish", image=image)
hf_cache = modal.Volume.from_name("scrubdata-hf-cache", create_if_missing=True)
results = modal.Dict.from_name("scrubdata-jellyfish-results", create_if_missing=True)

KEYWORDS = {"hospital": "hospital", "beers": "beer", "flights": "flight",
            "rayyan": "bibliography", "movies_1": "movie"}


@app.function(gpu="A100-80GB", timeout=4 * 3600, volumes={"/hf": hf_cache})
def run_jellyfish(model_id: str = "NECOUDBFM/Jellyfish-13B",
                  datasets: str = "hospital,beers,flights,rayyan,movies_1"):
    import os
    import sys
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    os.environ["HF_HOME"] = "/hf"
    from vllm import LLM, SamplingParams

    from eval.baselines_learned import di_prompt, ed_prompt, parse_di, parse_ed
    from eval.run_real_multi import _raha_pair, score

    llm = LLM(model=model_id, dtype="bfloat16", download_dir="/hf")
    # card-recommended decoding; stop must stay "### Instruction:"
    ed_params = SamplingParams(temperature=0.35, top_p=0.9, repetition_penalty=1.15,
                               max_tokens=6, stop=["### Instruction:"])
    di_params = SamplingParams(temperature=0.35, top_p=0.9, repetition_penalty=1.15,
                               max_tokens=64, stop=["### Instruction:"])

    out = {}
    for name in datasets.split(","):
        dirty, clean = _raha_pair(name)
        records = dirty.to_dict(orient="records")
        cells = [(i, col) for i, rec in enumerate(records) for col in dirty.columns]
        print(f"{name}: {len(cells)} ED prompts", flush=True)
        ed_out = llm.generate([ed_prompt(records[i], col) for i, col in cells], ed_params)
        flagged = [(i, col) for (i, col), o in zip(cells, ed_out)
                   if parse_ed(o.outputs[0].text)]
        print(f"{name}: {len(flagged)} flagged -> DI", flush=True)
        kw = KEYWORDS.get(name, "data")
        di_out = llm.generate([di_prompt(records[i], col, kw) for i, col in flagged],
                              di_params)
        repaired = dirty.copy()
        for (i, col), o in zip(flagged, di_out):
            repaired.loc[i, col] = parse_di(o.outputs[0].text, str(dirty.loc[i, col]))
        m = score(dirty, clean, repaired)
        out[name] = {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                     "damage": m["damage"], "n_flagged": len(flagged),
                     "n_cells": len(cells)}
        results[f"{model_id.rsplit('/', 1)[-1]}:{name}"] = {
            **out[name], "repaired_csv": repaired.to_csv(index=False)}
        print(f"  {name}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"dmg={m['damage']:.3f}", flush=True)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    summary = {"system": f"Jellyfish ED+DI ({model_id})",
               "real_f1": mean(d["f1"] for d in out.values()),
               "damage": mean(d["damage"] for d in out.values()),
               "precision": mean(d["precision"] for d in out.values()),
               "recall": mean(d["recall"] for d in out.values()),
               "per_dataset": out}
    results[f"{model_id.rsplit('/', 1)[-1]}:summary"] = summary
    print("\nJELLYFISH summary:", {k: round(v, 3) for k, v in summary.items()
                                   if isinstance(v, float)})
    return summary


@app.local_entrypoint()
def main(model_id: str = "NECOUDBFM/Jellyfish-13B",
         datasets: str = "hospital,beers,flights,rayyan,movies_1"):
    call = run_jellyfish.spawn(model_id=model_id, datasets=datasets)
    print(f"Launched detached. call_id={call.object_id}")
