"""Run the v4 fine-tune eval on a Modal GPU — fast, unlike local Q8 (250s/call timeouts).

Loads base + the v4 LoRA adapter in bf16 (better fidelity than the GGUF), runs both eval
layers (synthetic matrix + real hospital, batched). Cost-bounded: L4 GPU, ~15 min.

    uv run modal run scripts/modal_eval.py            # default n=20 synthetic
    uv run modal run scripts/modal_eval.py --n 12
"""

import modal

IGNORE = [".venv", ".git", "data", "*.gguf", "**/__pycache__", ".gstack",
          "frontend/variant_a", "frontend/variant_b", "frontend/variant_c"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate",
                 "pandas", "jsonschema", "huggingface_hub", "pycountry", "sentencepiece")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
)

app = modal.App("scrubdata-eval", image=image)


@app.function(gpu="L4", timeout=1800)
def run_eval(n_synth: int = 20):
    import os, sys
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    import torch
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
    adapter_id = "ricalanis/scrubdata-qwen3-4b-v4"
    tok = AutoTokenizer.from_pretrained(adapter_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_id).eval()

    def base_planner(df, *_):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(profile_dataframe(df), df)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=2000, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        plan = _extract_json(text)
        if plan is None:
            return {"__error__": "no_json"}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan

    out = {}
    # Layer 1 — synthetic frozen gold
    gold = load_gold()[:n_synth]
    systems = {"ORACLE": lambda df, gp: gp,
               "HEURISTIC": lambda df, gp: mock_plan(df),
               "FT_v4": base_planner}
    out["layer1"] = {name: evaluate(fn, gold) for name, fn in systems.items()}

    # Layer 2 — real hospital (batched)
    _ensure_data()
    dirty, clean = _load()
    ft_plan = make_batched_planner(base_planner, batch_size=6)(dirty)
    cleaned, _ = apply_plan(dirty, ft_plan)
    out["layer2_ft"] = _score(dirty, clean, cleaned)
    out["layer2_noop"] = _score(dirty, clean, dirty)
    return out


@app.local_entrypoint()
def main(n: int = 20):
    r = run_eval.remote(n_synth=n)
    print("\n=== Layer 1 (synthetic) ===")
    cols = ["json_valid", "op_f1", "canon_f1", "canon_r", "recovery"]
    print(f"{'system':<12}" + "".join(f"{c:>11}" for c in cols))
    for name, m in r["layer1"].items():
        print(f"{name:<12}" + "".join(f"{m[c]:>11.3f}" for c in cols))
    print("\n=== Layer 2 (real hospital) ===")
    for k in ("layer2_noop", "layer2_ft"):
        m = r[k]
        print(f"{k:<12} repair_recall={m['repair_recall']:.3f} "
              f"repair_prec={m['repair_prec']:.3f} recovery={m['recovery']:.3f} "
              f"fixed={m['_fixed']}/{m['_errors']}")
