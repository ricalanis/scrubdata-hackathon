"""QLoRA fine-tune of the ScrubData planner — Colab Pro+ (A100) or HF Jobs.

Trains a ≤4B model (default Qwen3-4B-Instruct-2507) on our verified SFT data to emit
the JSON cleaning plan reliably + in our conventions, then exports a Q4_K_M GGUF for
llama.cpp and pushes both adapter and GGUF to the Hub.

Recipe per project research (memory: training-recipe): A100/L4 → 16-bit LoRA;
r=32, alpha=32, all 7 target modules; LR 2e-4, 2-3 epochs, bf16. On a small GPU it
auto-falls back to 4-bit QLoRA.

Run (Colab, after the 3 setup cells in notebooks/README.md):
    !python notebooks/train_qlora.py \
        --data-repo build-small-hackathon/scrubdata-sft \
        --out-repo  <your-user>/scrubdata-qwen3-4b
HF_TOKEN must be set in the environment (Colab Secrets / `os.environ`).
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data-repo", default="build-small-hackathon/scrubdata-sft",
                    help="HF dataset repo holding train.jsonl (messages format)")
    ap.add_argument("--data-file", default="train.jsonl")
    ap.add_argument("--out-repo", default=None, help="HF repo to push adapter + GGUF")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq", type=int, default=6144,
                    help="v3 examples reach ~5.5k tokens; keep ≥6144 to avoid truncation")
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ.get("HF_TOKEN")

    # Big GPU (≈24GB+) → 16-bit LoRA (quality edge); small GPU → 4-bit QLoRA.
    vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    load_in_4bit = vram < 22e9
    big = not load_in_4bit
    print(f"GPU VRAM={vram/1e9:.0f}GB → {'16-bit LoRA' if big else '4-bit QLoRA'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq,
        load_in_4bit=load_in_4bit, full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)

    ds = load_dataset(args.data_repo, data_files=args.data_file, split="train")

    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(fmt, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=args.max_seq,
            # smaller batch since sequences are long now (~6k); effective batch stays 16
            per_device_train_batch_size=4 if big else 1,
            gradient_accumulation_steps=4 if big else 16,
            warmup_steps=5, num_train_epochs=args.epochs, learning_rate=2e-4,
            logging_steps=10, optim="adamw_8bit", weight_decay=0.001,
            lr_scheduler_type="linear", seed=0, bf16=big, fp16=not big,
            output_dir="outputs", report_to="none"))

    # Train only on the assistant's plan (mask the prompt) for cleaner SFT.
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n")
    except Exception as e:  # template markers vary by base model — non-fatal
        print(f"(train_on_responses_only skipped: {e})")

    trainer.train()

    out = "scrubdata-qwen3-4b"
    model.save_pretrained_gguf(out, tokenizer, quantization_method="q4_k_m")
    print(f"Saved GGUF under ./{out}")

    if args.out_repo and hf_token:
        model.push_to_hub(args.out_repo, token=hf_token)
        tokenizer.push_to_hub(args.out_repo, token=hf_token)
        model.push_to_hub_gguf(f"{args.out_repo}-gguf", tokenizer,
                               quantization_method="q4_k_m", token=hf_token)
        print(f"Pushed adapter → {args.out_repo} and GGUF → {args.out_repo}-gguf")
    else:
        print("Set --out-repo and HF_TOKEN to push artifacts to the Hub.")


if __name__ == "__main__":
    main()
