"""Evaluate the GROUNDED fine-tuned model on the wide validation suite (corrected,
churn-neutral metric) — the model row for the paper's money table.

Loads the v5 adapter (Modal volume) merged into bf16, wraps it batched + RACOON-grounded
(make_grounded_planner), and runs the suite's REAL slice (5 Raha benchmarks) plus the
typo-injected slice (one per harvested domain) — the canonicalization regime the model is
for. Single seed (the CI row comes from the cheap heuristic systems); scoped honestly.

    uv run modal run --detach scripts/modal_eval_suite.py
"""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**", "data/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.45", "peft", "accelerate",
                 "pandas", "jsonschema", "pycountry", "sentencepiece")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
    .add_local_dir("data/real/cache", "/root/repo/data/real/cache", copy=True)
    .add_local_file("training/unpaired_sources.json",
                    "/root/repo/training/unpaired_sources.json", copy=True)
)
app = modal.App("scrubdata-eval-suite", image=image)
adapter_vol = modal.Volume.from_name("scrubdata-v5-adapter")
results = modal.Dict.from_name("scrubdata-suite-results", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=7200, volumes={"/vol": adapter_vol})
def run_suite(seed: int = 7):
    import os, sys, torch
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
    from scrubdata.profiler import profile_dataframe
    from scrubdata.model_planner import _extract_json, make_batched_planner
    from scrubdata.grounded import make_grounded_planner
    from scrubdata.executor import apply_plan
    from eval.run_real_multi import build_suite, score, _cell_only, abstain_slice

    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, "/vol/v5").merge_and_unload()
    model.eval()
    model.config.use_cache = True
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id, im_end] if im_end is not None else tok.eos_token_id

    def base_planner(df, *_):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(profile_dataframe(df), df)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=enc["attention_mask"].to(model.device),
                                 max_new_tokens=2000, do_sample=False, eos_token_id=eos_ids,
                                 pad_token_id=tok.eos_token_id, use_cache=True,
                                 suppress_tokens=[151657, 151658])
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        plan = _extract_json(text)
        if plan is None:
            return {"__error__": "no_json"}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan

    planner = make_grounded_planner(make_batched_planner(base_planner, batch_size=4))

    # scoped slice: all REAL + the typo-injected datasets (the canonicalization regime)
    specs = [s for s in build_suite(seed=seed)
             if s.get("source") == "real" or s["name"].endswith(":typo")]
    rows = []
    for spec in specs:
        try:
            loaded = spec["load"]()
        except Exception as e:  # noqa: BLE001
            print(f"  {spec['name']}: load failed {type(e).__name__}", flush=True)
            continue
        if loaded is None:
            continue
        dirty, clean = loaded
        try:
            cleaned, _ = apply_plan(dirty, _cell_only(planner(dirty)))
            m = score(dirty, clean, cleaned)
        except Exception as e:  # noqa: BLE001
            print(f"  {spec['name']}: eval failed {type(e).__name__}", flush=True)
            continue
        rows.append({"name": spec["name"], "source": spec.get("source", "injected"),
                     "f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
                     "damage": m["damage"]})
        print(f"  {spec['name']:<26} F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} dmg={m['damage']:.3f}", flush=True)

    ab = abstain_slice(planner)
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0
    summary = {
        "real_f1": mean(r["f1"] for r in rows if r["source"] == "real"),
        "injected_typo_f1": mean(r["f1"] for r in rows if r["source"] != "real"),
        "damage": mean(r["damage"] for r in rows),
        "abstain_accuracy": ab["abstain_accuracy"], "typo_recall": ab["typo_recall"],
        "n_datasets": len(rows), "rows": rows,
    }
    print("\nGROUNDED MODEL (v5+RACOON) on suite:", {k: round(v, 3) for k, v in summary.items()
                                                    if isinstance(v, float)})
    results["latest"] = summary
    return summary


@app.local_entrypoint()
def main():
    call = run_suite.spawn()
    print(f"Launched detached. call_id={call.object_id}")
