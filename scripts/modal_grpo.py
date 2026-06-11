"""GRPO pilot: RL the 4B planner against OUR EXECUTOR as the verifiable reward.

The published recipe (Table-R1, Spreadsheet-RL-4B, SFT-memorizes/RL-generalizes)
applied with the asset those papers had to build and we already own: a deterministic
executor that scores any plan against recoverable gold. Reward per completion:

    invalid JSON ........................ -1.0
    + 0.2 valid JSON, + 0.2 schema-valid (composite partial rewards per SQL-R1)
    + 2.0 * churn-neutral F1 on the episode window
    - 4.0 * damage rate (the never-corrupt-clean-data contract, in the loss)

CONTROL ARM (mandatory; Spurious Rewards): --control trains with random rewards on
identical config — gains must beat the control to count.

Budget guards: hard timeout (the only real cost driver), step cap, single A100-80GB.

    uv run modal run --detach scripts/modal_grpo.py                    # main pilot
    uv run modal run --detach scripts/modal_grpo.py --control          # control arm
"""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**",
          "data/**", "eval/results/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.45", "peft", "accelerate", "trl>=0.14",
                 "datasets", "pandas", "jsonschema", "pycountry", "sentencepiece")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
    .add_local_file("data/grpo_episodes.jsonl", "/root/repo/data/grpo_episodes.jsonl",
                    copy=True)
)
app = modal.App("scrubdata-grpo", image=image)
results = modal.Dict.from_name("scrubdata-train-results", create_if_missing=True)
adapter_vol = modal.Volume.from_name("scrubdata-v5-adapter")


@app.function(gpu="A100-80GB", timeout=4 * 3600, volumes={"/vol": adapter_vol})
def train_grpo(steps: int = 150, control: bool = False, seed: int = 0,
               num_generations: int = 6, lr: float = 1e-5):
    import io
    import json
    import os
    import random
    import sys

    import torch
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    import pandas as pd
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from eval.metrics import is_valid
    from eval.run_real_multi import _cell_only, score
    from scrubdata.executor import apply_plan
    from scrubdata.model_planner import _extract_json

    episodes = [json.loads(l) for l in open("data/grpo_episodes.jsonl")]
    random.Random(seed).shuffle(episodes)
    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(base_id)

    rows = []
    for ep in episodes:
        prompt = tok.apply_chat_template(ep["messages"], tokenize=False,
                                         add_generation_prompt=True)
        rows.append({"prompt": prompt, "dirty_csv": ep["dirty_csv"],
                     "clean_csv": ep["clean_csv"]})
    ds = Dataset.from_list(rows)
    rng = random.Random(seed + 1)

    def reward_fn(completions, dirty_csv, clean_csv, **kw):
        out = []
        for comp, dcsv, ccsv in zip(completions, dirty_csv, clean_csv):
            if control:
                out.append(rng.random())          # Spurious-Rewards control arm
                continue
            plan = _extract_json(comp)
            if plan is None:
                out.append(-1.0)
                continue
            r = 0.2
            plan.setdefault("table_operations", [])
            plan.setdefault("columns", [])
            plan.setdefault("flags", [])
            if is_valid(plan):
                r += 0.2
            try:
                dirty = pd.read_csv(io.StringIO(dcsv), dtype=str, keep_default_na=False)
                clean = pd.read_csv(io.StringIO(ccsv), dtype=str, keep_default_na=False)
                cleaned, _ = apply_plan(dirty, _cell_only(plan))
                m = score(dirty, clean, cleaned)
                r += 2.0 * m["f1"] - 4.0 * m["damage"]
            except Exception:  # noqa: BLE001
                r -= 0.5                          # plan executed badly: penalize
            out.append(r)
        return out

    cfg = GRPOConfig(
        output_dir="/tmp/grpo",
        per_device_train_batch_size=num_generations,    # one prompt group / device step
        num_generations=num_generations,
        gradient_accumulation_steps=2,
        max_steps=steps,
        learning_rate=lr,
        max_prompt_length=2304,
        max_completion_length=1024,
        temperature=0.9,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=base_id,
        args=cfg,
        reward_funcs=reward_fn,
        train_dataset=ds,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj",
                                                                    "v_proj", "o_proj"],
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    dest = "/vol/grpo_control" if control else "/vol/grpo_pilot"
    trainer.model.save_pretrained(dest)
    adapter_vol.commit()
    hist = [h for h in trainer.state.log_history if "reward" in h]
    summary = {"arm": "control" if control else "main", "steps": steps,
               "reward_curve": [(h.get("step"), round(h.get("reward", 0), 3))
                                for h in hist][-15:],
               "adapter": dest}
    results[f"grpo_{'control' if control else 'pilot'}"] = summary
    print("GRPO DONE:", summary)
    return summary


@app.local_entrypoint()
def main(steps: int = 150, control: bool = False):
    call = train_grpo.spawn(steps=steps, control=control)
    print(f"Launched detached. call_id={call.object_id}")
