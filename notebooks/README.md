# Training on Colab Pro+ (start a run in ~3 cells)

Fine-tunes the ScrubData planner (Qwen3-4B-Instruct-2507) with QLoRA, exports a GGUF,
and pushes to the Hub. ~1–2h on an A100; Pro+ background execution finishes it unattended.

## One-time prep (on this machine)
```bash
uv run training/build_dataset.py --n 2000 --out data/train.jsonl   # verified SFT data
huggingface-cli login                                              # HF token with WRITE
uv run scripts/push_dataset.py --repo <you>/scrubdata-sft          # data -> HF dataset
```

## In Colab
1. **New notebook → Runtime → Change runtime type → A100 GPU + High-RAM.** (Fallback L4;
   T4 works too — the script auto-switches to 4-bit there.)
2. Add `HF_TOKEN` in Colab **Secrets** (🔑), value = your HF write token.
3. Run these cells:

**Cell 1 — deps + repo**
```python
!pip install -q unsloth
!git clone https://github.com/<you>/hackaton-small.git || true   # or upload notebooks/train_qlora.py
```

**Cell 2 — token**
```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

**Cell 3 — train + export + push**
```python
!python hackaton-small/notebooks/train_qlora.py \
    --data-repo <you>/scrubdata-sft \
    --out-repo  <you>/scrubdata-qwen3-4b \
    --epochs 2
```

Artifacts land at `https://huggingface.co/<you>/scrubdata-qwen3-4b` (adapter) and
`...-gguf` (llama.cpp). The Colab VM is ephemeral — the script pushes to the Hub so a
finished run isn't lost.

## After training — measure it
Pull the GGUF locally and plug it into the eval harness as a planner (same shape as
`scrubdata/model_planner.py`), then:
```bash
uv run eval/run_model.py --model <local-gguf-or-ollama-id>   # vs heuristic/oracle
uv run eval/run_real.py                                       # real OOD slice
```
Compare against the goalposts in `eval/README.md` (recovery ≥0.95, canon_f1 ≥0.85).

## Can Claude start this for me?
Not headlessly — Google removed external Colab runtime control and tunneling violates
ToS. Options: (a) run the 3 cells yourself (Pro+ background execution carries it), or
(b) connect **colab-mcp** with a Colab tab open and Claude Code can drive the cells live
in your browser (co-pilot, not unattended). HF Jobs is the headless alternative if you
ever want a no-browser run.
