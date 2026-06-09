"""Fast standalone eval of the v5 adapter saved in the Modal Volume.

The in-training eval was slow because it ran the 4-bit (bitsandbytes) training model.
Here we load the base in BF16 + the LoRA adapter from the volume and merge -> fast
generation. Reports synthetic gold + the real Raha hospital repair_recall (the headline).

    uv run modal run scripts/modal_eval_v5.py
"""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**", "data/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.45", "peft", "accelerate",
                 "pandas", "jsonschema", "pycountry", "sentencepiece")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
)
app = modal.App("scrubdata-eval-v5", image=image)
adapter_vol = modal.Volume.from_name("scrubdata-v5-adapter")
results = modal.Dict.from_name("scrubdata-eval-v5-results", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=2400, volumes={"/vol": adapter_vol})
def run_eval(n_synth: int = 20):
    import os, sys, torch
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
    from scrubdata.profiler import profile_dataframe
    from scrubdata.model_planner import _extract_json, make_batched_planner
    from scrubdata.executor import apply_plan
    from scrubdata.planner import mock_plan
    from eval.run_eval import evaluate
    from eval.gold import load_gold
    from eval.run_real import _ensure_data, _load, _score

    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, "/vol/v5").merge_and_unload()  # bf16-native merge
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
                                 max_new_tokens=2200, do_sample=False, eos_token_id=eos_ids,
                                 pad_token_id=tok.eos_token_id, use_cache=True,
                                 suppress_tokens=[151657, 151658])  # block <tool_call> loop
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        plan = _extract_json(text)
        if plan is None:
            return {"__error__": "no_json"}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan

    out = {}
    gold = load_gold()[:n_synth]
    out["layer1"] = {name: evaluate(fn, gold) for name, fn in {
        "HEURISTIC": lambda df, gp: mock_plan(df), "FT_v5": base_planner}.items()}
    _ensure_data()
    dirty, clean = _load()
    ft_plan = make_batched_planner(base_planner, batch_size=4)(dirty)
    cleaned, _ = apply_plan(dirty, ft_plan)
    out["hospital_ft"] = _score(dirty, clean, cleaned)
    out["hospital_noop"] = _score(dirty, clean, dirty)

    table = _format(out)
    print(table)
    results["latest"] = {"out": out, "table": table}
    return out


def _format(r) -> str:
    L = ["\n=== Layer 1 (synthetic) ==="]
    cols = ["json_valid", "op_f1", "canon_f1", "recovery"]
    L.append(f"{'system':<12}" + "".join(f"{c:>11}" for c in cols))
    for name, m in r["layer1"].items():
        L.append(f"{name:<12}" + "".join(f"{m[c]:>11.3f}" for c in cols))
    L.append("\n=== Real hospital ===")
    for k in ("hospital_noop", "hospital_ft"):
        m = r[k]
        L.append(f"{k:<13} repair_recall={m['repair_recall']:.3f} "
                 f"repair_prec={m['repair_prec']:.3f} recovery={m['recovery']:.3f}")
    return "\n".join(L)


@app.local_entrypoint()
def main():
    call = run_eval.spawn()
    print(f"Launched detached. call_id={call.object_id}")
