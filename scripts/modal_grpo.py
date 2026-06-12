"""GRPO pilot: RL the 4B planner against OUR EXECUTOR as the verifiable reward.

Hand-rolled GRPO loop (TRL 0.14-0.17 all hard-require vllm at GRPO import in this
stack; the algorithm is ~100 lines and the pilot question is signal, not framework
purity): per step, sample G completions for one episode prompt, reward each by
EXECUTING the plan against the episode's clean slice, normalize advantages within
the group, take a policy-gradient step on LoRA params. No KL-ref term in the pilot
(LoRA r16 + lr 1e-5 bounds drift; disclosed).

Reward: invalid JSON -1.0; +0.2 valid JSON; +0.2 schema-valid;
        +2.0 * churn-neutral F1 − 4.0 * damage; execution exception −0.5.
CONTROL ARM (--control): random rewards, identical config (Spurious Rewards check).

    uv run modal run --detach scripts/modal_grpo.py                 # main, 150 steps
    uv run modal run --detach scripts/modal_grpo.py --control --steps 100
"""

import modal

IGNORE = [".venv/**", ".git/**", "*.gguf", "**/__pycache__/**", ".gstack/**",
          "design/**", "frontend/variant_*/**", "notebooks/**", ".pytest_cache/**",
          "data/**", "eval/results/**"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=4.45", "peft", "accelerate",
                 "pandas", "jsonschema", "pycountry", "sentencepiece")
    .add_local_dir(".", "/root/repo", ignore=IGNORE, copy=True)
    .add_local_file("data/grpo_episodes.jsonl", "/root/repo/data/grpo_episodes.jsonl",
                    copy=True)
)
app = modal.App("scrubdata-grpo", image=image)
results = modal.Dict.from_name("scrubdata-train-results", create_if_missing=True)
adapter_vol = modal.Volume.from_name("scrubdata-v5-adapter")


@app.function(gpu="A100-80GB", timeout=4 * 3600, volumes={"/vol": adapter_vol})
def train_grpo(steps: int = 150, control: bool = False, seed: int = 0,
               group: int = 6, lr: float = 5e-6, max_new: int = 1024,
               kl_beta: float = 0.05, dest_name: str = ""):
    import io
    import json
    import os
    import random
    import sys

    import torch
    os.chdir("/root/repo")
    sys.path.insert(0, "/root/repo")
    import pandas as pd
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval.metrics import is_valid
    from eval.run_real_multi import _cell_only, score
    from scrubdata.executor import apply_plan
    from scrubdata.model_planner import _extract_json

    torch.manual_seed(seed)
    rng = random.Random(seed)
    episodes = [json.loads(l) for l in open("data/grpo_episodes.jsonl")]
    rng.shuffle(episodes)

    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])

    def reward(comp: str, ep) -> float:
        if control:
            return rng.random()
        plan = _extract_json(comp)
        if plan is None:
            return -1.0
        r = 0.2
        plan.setdefault("table_operations", [])
        plan.setdefault("columns", [])
        plan.setdefault("flags", [])
        if is_valid(plan):
            r += 0.2
        try:
            dirty = pd.read_csv(io.StringIO(ep["dirty_csv"]), dtype=str,
                                keep_default_na=False)
            clean = pd.read_csv(io.StringIO(ep["clean_csv"]), dtype=str,
                                keep_default_na=False)
            cleaned, _ = apply_plan(dirty, _cell_only(plan))
            m = score(dirty, clean, cleaned)
            r += 2.0 * m["f1"] - 4.0 * m["damage"]
        except Exception:  # noqa: BLE001
            r -= 0.5
        return r

    curve = []
    for step in range(steps):
        ep = episodes[step % len(episodes)]
        prompt = tok.apply_chat_template(ep["messages"], tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2304)
        ids = enc["input_ids"].cuda()
        attn = enc["attention_mask"].cuda()
        with torch.no_grad():
            gen = model.generate(input_ids=ids.repeat(group, 1),
                                 attention_mask=attn.repeat(group, 1),
                                 do_sample=True, temperature=0.9, top_p=0.95,
                                 max_new_tokens=max_new, eos_token_id=eos_ids,
                                 pad_token_id=tok.eos_token_id,
                                 suppress_tokens=[151657, 151658])
        plen = ids.shape[1]
        comps = [tok.decode(g[plen:], skip_special_tokens=True) for g in gen]
        rs = torch.tensor([reward(c, ep) for c in comps], dtype=torch.float32)
        mean_r = rs.mean().item()
        curve.append((step, round(mean_r, 3)))
        if float(rs.std()) < 1e-5:
            continue                                  # degenerate group: no signal
        adv = (rs - rs.mean()) / (rs.std() + 1e-6)
        # teacher-forced logprobs of sampled completions under the current policy
        opt.zero_grad()
        loss_total = 0.0
        for g_seq, a in zip(gen, adv.tolist()):
            if abs(a) < 1e-6:
                continue
            seq = g_seq.unsqueeze(0)
            # completion-token labels: mask the prompt and everything after the
            # first eos in the completion region
            labels = seq.clone()
            labels[:, :plen] = -100
            comp_region = seq[:, plen:]
            eos_pos = (comp_region == tok.eos_token_id) | \
                      ((comp_region == im_end) if im_end is not None else
                       torch.zeros_like(comp_region, dtype=torch.bool))
            after_first_eos = eos_pos.float().cumsum(dim=1) > 1
            labels[:, plen:][after_first_eos] = -100
            out = model(input_ids=seq)
            logits = out.logits[:, :-1]
            tgt = labels[:, 1:]
            mask = tgt != -100
            lp = torch.log_softmax(logits.float(), dim=-1)
            tok_lp = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            mean_lp = (tok_lp * mask).sum() / mask.sum().clamp(min=1)
            # KL anchor to the frozen base (v2 fix: v1 ran unanchored and BOTH
            # arms destroyed JSON discipline — pure RL drift, caught by the
            # random-reward control). With LoRA the ref is free: disable adapters.
            kl = torch.tensor(0.0, device=seq.device)
            if kl_beta > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref_logits = model(input_ids=seq).logits[:, :-1]
                    ref_lp_tok = torch.log_softmax(ref_logits.float(), dim=-1).gather(
                        -1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                kl = ((tok_lp - ref_lp_tok) * mask).sum() / mask.sum().clamp(min=1)
            loss = (-(a * mean_lp) + kl_beta * kl.abs()) / group
            loss.backward()
            loss_total += float(loss)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 5 == 0:
            recent = [r for _, r in curve[-10:]]
            print(f"step {step}: reward {mean_r:.3f} (avg-10 "
                  f"{sum(recent)/len(recent):.3f}) loss {loss_total:.4f}", flush=True)

    dest = dest_name or ("/vol/grpo_control" if control else "/vol/grpo_pilot")
    model.save_pretrained(dest)
    adapter_vol.commit()
    n25 = min(25, len(curve))
    summary = {"arm": "control" if control else "main", "steps": steps,
               "reward_first10": curve[:10], "reward_last10": curve[-10:],
               "reward_mean_first25": round(sum(r for _, r in curve[:n25]) / n25, 3),
               "reward_mean_last25": round(sum(r for _, r in curve[-n25:]) / n25, 3),
               "adapter": dest}
    key = dest.rsplit("/", 1)[-1]
    results[key] = summary
    print("GRPO DONE:", summary)
    return summary


@app.local_entrypoint()
def main(steps: int = 150, control: bool = False, dest_name: str = ""):
    call = train_grpo.spawn(steps=steps, control=control, dest_name=dest_name)
    print(f"Launched detached. call_id={call.object_id}")
