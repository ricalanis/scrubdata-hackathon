"""Minimal: load the v5 adapter, generate ONE plan, print the RAW output to see why
json_valid is 0 (despite train_loss 0.16). Fast (~4 min)."""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**", "data/**"]
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers>=4.45", "peft", "accelerate", "pandas",
                      "jsonschema", "pycountry", "sentencepiece")
         .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True))
app = modal.App("scrubdata-debug-v5", image=image)
vol = modal.Volume.from_name("scrubdata-v5-adapter")
dbg = modal.Dict.from_name("scrubdata-debug-v5", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=900, volumes={"/vol": vol})
def debug():
    import os, sys, torch
    os.chdir("/root/repo"); sys.path.insert(0, "/root/repo")
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
    from scrubdata.profiler import profile_dataframe
    from scrubdata.model_planner import _extract_json

    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, "/vol/v5").merge_and_unload()
    model.eval()

    df = pd.DataFrame({"country": ["USA", "usa", "U.S.A", "USA", "Canada", "canada"],
                       "status": ["Won", "won", "WON", "Lost", "lost", "Won"]})
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(profile_dataframe(df), df)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(model.device)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    # suppress the <tool_call>/</tool_call> special tokens (151657/151658) that the
    # base model loops on, forcing it to its learned next-best token ('{').
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=enc["attention_mask"].to(model.device),
                             max_new_tokens=900, do_sample=False,
                             eos_token_id=[tok.eos_token_id, im_end], pad_token_id=tok.eos_token_id,
                             suppress_tokens=[151657, 151658])
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    plan = _extract_json(raw)
    canon = None
    if isinstance(plan, dict):
        canon = {k: v for c in plan.get("columns", []) for o in c.get("operations", [])
                 if o.get("op") == "canonicalize_categories" for k, v in o.get("mapping", {}).items()}
    info = {"raw_len": len(raw), "raw_head": raw[:500],
            "extracted_keys": list(plan.keys()) if isinstance(plan, dict) else None,
            "canon": canon}
    dbg["latest"] = info
    print(info)
    return info


@app.local_entrypoint()
def main():
    print(debug.remote())
