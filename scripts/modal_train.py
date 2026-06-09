"""Train ScrubData v5 (QLoRA on synthetic high-card + real enriched data) on a Modal
GPU, then eval on synthetic gold + the real Raha hospital table — in one shot.

Standard HF stack (bitsandbytes 4-bit + peft LoRA + Trainer) for robustness. The
trained adapter stays in-GPU for eval, so NO HF token / push is needed. The headline
number: does hospital repair_recall finally clear 0 after training on real data?

    uv run modal run scripts/modal_train.py                 # 2 epochs
    uv run modal run scripts/modal_train.py --epochs 3
"""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**",
          "data/**"]   # exclude all data; add just the v5 training file below

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.45", "peft", "accelerate", "bitsandbytes",
                 "datasets", "pandas", "jsonschema", "pycountry", "sentencepiece")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
    .add_local_file("data/v5_train.jsonl", "/root/repo/data/v5_train.jsonl", copy=True)
)
app = modal.App("scrubdata-train", image=image)
results = modal.Dict.from_name("scrubdata-train-results", create_if_missing=True)
adapter_vol = modal.Volume.from_name("scrubdata-v5-adapter", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=5400, volumes={"/vol": adapter_vol})
def train_and_eval(epochs: int = 1, max_len: int = 2560, lr: float = 2e-4, n_synth: int = 8):
    import os, sys, json, torch
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(base_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # BF16-native (NOT 4-bit): the adapter then matches a bf16 base exactly, so
    # merge_and_unload is clean (no quant mismatch -> no degenerate outputs) and
    # merged inference is fast. A100-80GB fits a 4B bf16 + LoRA easily.
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # ---- data: mask the prompt, train only on the assistant JSON plan ----
    def encode(msgs):
        # render to STRING first then tokenize (apply_chat_template(tokenize=True)
        # returns a nested list on this tokenizer -> len()==1, breaks masking).
        full_s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_s = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok(full_s, add_special_tokens=False)["input_ids"]
        prompt = tok(prompt_s, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt) + full[len(prompt):]
        return full[:max_len], labels[:max_len]

    data = []
    for line in open("data/v5_train.jsonl"):
        ids, lab = encode(json.loads(line)["messages"])
        if len(ids) >= 8 and any(t != -100 for t in lab):
            data.append({"input_ids": ids, "labels": lab})
    print(f"[train] {len(data)} examples, max_len={max_len}")

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, i): return data[i]

    def collate(batch):
        ml = max(len(b["input_ids"]) for b in batch)
        ii, ll, am = [], [], []
        for b in batch:
            pad = ml - len(b["input_ids"])
            ii.append(b["input_ids"] + [tok.pad_token_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            am.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}

    args = TrainingArguments(
        output_dir="/tmp/out", per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=25, save_strategy="no", report_to=[], optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
    train_out = trainer.train()
    final_loss = float(train_out.training_loss)
    print(f"\n[train] *** DONE, train_loss={final_loss:.4f} ***\n")

    # durability: persist the adapter BEFORE eval.
    model.save_pretrained("/vol/v5")
    adapter_vol.commit()
    print("[train] adapter saved to volume scrubdata-v5-adapter:/v5")

    # ---- eval: disable checkpointing (KV cache) + MERGE the bf16-native adapter for
    # fast, correct inference.
    model.gradient_checkpointing_disable()
    model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    from scrubdata.prompt import SYSTEM_PROMPT, build_user_prompt
    from scrubdata.profiler import profile_dataframe
    from scrubdata.model_planner import _extract_json, make_batched_planner
    from scrubdata.executor import apply_plan
    from scrubdata.planner import mock_plan
    from eval.run_eval import evaluate
    from eval.gold import load_gold
    from eval.run_real import _ensure_data, _load, _score

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
                                 max_new_tokens=1500, do_sample=False, eos_token_id=eos_ids,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        plan = _extract_json(text)
        if plan is None:
            return {"__error__": "no_json"}
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        return plan

    out = {"train_loss": final_loss}
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
    L = [f"\n[train_loss] {r['train_loss']:.4f}", "\n=== Layer 1 (synthetic) ==="]
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
def main(epochs: int = 2):
    call = train_and_eval.spawn(epochs=epochs)
    print(f"Launched detached. call_id={call.object_id}")
    print("Fetch: uv run python -c \"import modal;"
          "print(modal.Dict.from_name('scrubdata-train-results')['latest']['table'])\"")
