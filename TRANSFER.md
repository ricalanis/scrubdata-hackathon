# Machine transfer guide

Everything needed to continue this project on a new machine.

## 1. Clone + deps
```bash
git clone https://github.com/ricalanis/scrubdata-hackathon.git ~/Dev/hackaton-small
cd ~/Dev/hackaton-small && uv sync
uv run pytest tests/   # 25 tests should pass
```

## 2. Restore Claude Code memory (IMPORTANT)
The agent's persistent memory is bundled in `project-memory/`. On the new machine, after
opening the project in Claude Code once (so the project dir exists):
```bash
cp project-memory/*.md ~/.claude/projects/-Users-<USER>-Dev-hackaton-small/memory/
```
(Adjust the path-keyed directory name to the new machine's project path. `MEMORY.md` is the
index; the rest are the knowledge base — data-loop-playbook.md and arxiv-paper.md are the
operational core.)

## 3. Cloud auth (state lives in the cloud, just re-authenticate)
```bash
uv run modal token new        # Modal: adapters in volume scrubdata-v5-adapter
                              #   (/v5 = v5, /v5_seed21 = v6/mixA winner, seeds 1-3,25,26)
                              # results Dicts: scrubdata-train-results (seedN keys),
                              #   scrubdata-eval-v5-results, scrubdata-suite-results
hf auth login                 # HF: Space build-small-hackathon/scrubdata, model repos
                              #   ricalanis/scrubdata-qwen3-4b{,-v6-q8}, traces dataset
gh auth login                 # GitHub
```

## 4. Local model (optional, 4.3GB)
```bash
ollama pull hf.co/ricalanis/scrubdata-qwen3-4b-v6-q8:Q8_0
ollama create scrubdata-ft-v6 -f notebooks/Modelfile
SCRUBDATA_MODEL=scrubdata-ft-v6 uv run server.py
```

## 5. Regenerable data (data/ is gitignored)
Harvested alias vocabularies + paired examples are PRESERVED in `training/harvests/` —
copy them back so the generator finds them:
```bash
mkdir -p data && cp training/harvests/*.jsonl data/
```
Big training mixes are regenerable:
```bash
uv run python -m training.build_dataset --n 1600 --out data/v5_synth.jsonl --seed 5
uv run python -m training.real_data --datasets hospital beers movies_1 --per-dataset 80 --out data/v6_paired_big.jsonl
# mix recipe (mixA = winner): synth + paired*4, shuffled -> data/v5_train.jsonl
```
The eval suite re-fetches Raha benchmarks automatically; harvested gov/GitHub CSVs
(data/real/cache) re-download via training/unpaired_sources.json.

## 6. In-flight at transfer time
- mixH (additive-composition test, seed 30): Modal call `fc-01KTRXTHJKW3G81BT4Q0FZET8G`,
  result lands in Dict `scrubdata-train-results` key `seed30`. Retrieve from any machine:
  ```bash
  uv run python -c "import modal; print(modal.Dict.from_name('scrubdata-train-results').get('seed30'))"
  ```
- Open question it answers: whether the vocab-mix regressions (mixE/F/G ~0.57-0.59 vs mixA
  0.748) were eval-coverage shift. See project-memory/data-loop-playbook.md.

## 7. Where everything lives
- Paper: `docs/paper/main.tex` (+ numbers.tex, fig) — compiles with pdflatex; COMPLETE.
- Submission kit: `docs/SUBMISSION.md` (demo script + social post), `docs/FIELD_NOTES.md`.
- Live Space: https://huggingface.co/spaces/build-small-hackathon/scrubdata
- arXiv next steps: cs.DB endorser etc. — project-memory/arxiv-paper.md.
- Hackathon deadline: 2026-06-15 (demo video + social post remain).
