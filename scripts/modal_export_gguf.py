"""Export the blessed v6 adapter (Modal volume) -> merged bf16 -> GGUF Q8_0 -> HF Hub.

Q8_0 only: Q4_K_M corrupts this model (documented v4 failure). CPU instance (merge +
convert are RAM-bound, no GPU needed). The HF token is passed as a function argument at
spawn time — transient, never baked into the image or logged.

    uv run modal run scripts/modal_export_gguf.py --repo ricalanis/scrubdata-qwen3-4b-v6-q8
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers>=4.45", "peft", "accelerate", "sentencepiece",
                 "huggingface_hub", "gguf", "numpy", "safetensors")
    .run_commands("git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp")
)
app = modal.App("scrubdata-export-gguf", image=image)
adapter_vol = modal.Volume.from_name("scrubdata-v5-adapter")


@app.function(cpu=8, memory=49152, timeout=3600, volumes={"/vol": adapter_vol})
def export(repo: str, hf_token: str, adapter: str = "/vol/v5_seed21"):
    import subprocess
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi

    base_id = "unsloth/Qwen3-4B-Instruct-2507"
    print("loading base + adapter (CPU, bf16)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_id)
    model.save_pretrained("/tmp/merged", safe_serialization=True)
    tok.save_pretrained("/tmp/merged")
    print("merged model saved", flush=True)

    out = "/tmp/scrubdata-qwen3-4b-v6.Q8_0.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", "/tmp/merged",
                    "--outfile", out, "--outtype", "q8_0"], check=True)
    import os
    print(f"GGUF ready: {os.path.getsize(out)/1e9:.2f} GB", flush=True)

    api = HfApi(token=hf_token)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=out, path_in_repo="scrubdata-qwen3-4b-v6.Q8_0.gguf",
                    repo_id=repo, repo_type="model",
                    commit_message="v6 (mixA): Q8_0 GGUF — hospital repair 0.475/0.185")
    print(f"uploaded to https://huggingface.co/{repo}", flush=True)
    return repo


@app.local_entrypoint()
def main(repo: str = "ricalanis/scrubdata-qwen3-4b-v6-q8"):
    from huggingface_hub import get_token
    token = get_token()
    assert token, "no local HF token found"
    call = export.spawn(repo=repo, hf_token=token)
    print(f"Launched detached. call_id={call.object_id}")
