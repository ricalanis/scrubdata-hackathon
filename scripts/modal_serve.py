"""Scale-to-zero Modal GPU endpoint serving the ScrubData v6 fine-tune via Ollama.

Mirrors Ollama's HTTP API so the repo's existing planner
(`scrubdata.model_planner.make_local_ollama_planner`) works UNCHANGED against the
public Modal URL: it POSTs {URL}/api/chat and reads response["message"]["content"].

Design:
  - GGUF (ricalanis/scrubdata-qwen3-4b-v6-q8, Q8_0) is downloaded INTO the image at
    BUILD time and the non-thinking Modelfile is written, so cold start does not
    re-download (only `ollama serve` boot + model load to GPU on first request).
  - `ollama create scrubdata-ft -f /Modelfile` runs at container start (fast: GGUF
    is already on local disk).
  - scale-to-zero via scaledown_window=300 -> $0 when idle; GPU cost only during use.

Deploy:
    uv run modal deploy scripts/modal_serve.py
    # -> public URL of the web_server (Ollama port 11434)
"""

import modal

HF_REPO = "ricalanis/scrubdata-qwen3-4b-v6-q8"
GGUF_FILE = "scrubdata-qwen3-4b-v6.Q8_0.gguf"
GGUF_PATH = f"/models/{GGUF_FILE}"
MODELFILE_PATH = "/models/Modelfile"
OLLAMA_PORT = 11434

# Non-thinking template — identical to notebooks/Modelfile. The Q8 GGUF must use the
# bare im_start/im_end chat template (no Qwen3 thinking/tools wrapper) or it burns the
# budget "thinking"; format=json in the API call grammar-constrains away the
# <tool_call> degeneration loop on long prompts.
MODELFILE = """FROM /models/scrubdata-qwen3-4b-v6.Q8_0.gguf
TEMPLATE \"\"\"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER repeat_penalty 1
PARAMETER top_k 20
PARAMETER top_p 0.95
"""


def _bake_model():
    """Build-time: pull GGUF from HF onto the image disk and write the Modelfile."""
    import os
    from huggingface_hub import hf_hub_download

    os.makedirs("/models", exist_ok=True)
    path = hf_hub_download(repo_id=HF_REPO, filename=GGUF_FILE, local_dir="/models")
    # hf_hub_download may symlink into a cache; ensure the literal path exists.
    if path != GGUF_PATH and not os.path.exists(GGUF_PATH):
        os.symlink(path, GGUF_PATH)
    with open(MODELFILE_PATH, "w") as f:
        f.write(MODELFILE)
    print(f"baked {GGUF_PATH} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd")
    # Pin Ollama 0.21.2: the REPAIRED Q8 GGUF was verified on 0.21.2 with format=json;
    # 0.30.7+ silently IGNORES format=json for this model and the planner degenerates
    # into <tool_call> loops (see eval/sc_rerank.py).
    .run_commands("curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.21.2 sh")
    .pip_install("huggingface_hub")
    .run_function(_bake_model)
)

app = modal.App("scrubdata-serve", image=image)


@app.function(
    gpu="A100",            # 40GB A100: ~2x prefill of A10G on our heavy 9k-token prompt
                           # (~95s -> ~50s/clean); model is ~4.7GB Q8 so 40GB is ample.
                           # scale-to-zero keeps idle cost $0; ~$0.05/clean active.
    scaledown_window=300,  # scale-to-zero ~5 min after last request -> $0 idle
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=OLLAMA_PORT, startup_timeout=300)
def serve():
    import subprocess
    import time
    import urllib.request

    env = {
        "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
        "OLLAMA_MODELS": "/root/.ollama/models",
        # Disable flash attention: the CUDA FA kernel path produced different decode
        # numerics than the CPU/desktop-GPU reference and let the <tool_call> token
        # leak past the format=json grammar constraint. Off => matches the verified
        # 0.21.2 reference behavior.
        "OLLAMA_FLASH_ATTENTION": "0",
        "OLLAMA_KV_CACHE_TYPE": "f16",
    }
    import os
    full_env = {**os.environ, **env}

    subprocess.Popen(["ollama", "serve"], env=full_env)

    # wait for the daemon
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(0.5)

    # create the named model from the baked GGUF (fast: local file)
    subprocess.run(["ollama", "create", "scrubdata-ft", "-f", MODELFILE_PATH],
                   env=full_env, check=True)
    print("scrubdata-ft created; serving", flush=True)
    # web_server keeps the process alive; ollama serve is already in the background.
