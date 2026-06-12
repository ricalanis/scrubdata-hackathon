# W1.c — ≤32B Zero-Label Repair Scaling Arm (multi-family, zero-shot)

First scaling measurement for the verified-union planner: vanilla (NOT fine-tuned)
20–31B open-weights models dropped into the EXACT hospital pipeline the 4B fine-tune
gate used — batched raw planner (batch_size=4, same `scrubdata/prompt.py` contract,
temperature 0) → `verify_plan(tau=0.5)` → union with the grounded heuristic
(`mock_plan`). Scored against hospital's 509 real errors with the
`eval/precision_curve.py` repairs-only churn-neutral protocol. Protocol parity was
verified by re-scoring the captured v6 plan through the same scorer: it reproduces the
prior gate numbers exactly (gated 0.993/0.287, union 0.905/0.413).

Disclosure: ≤32B open-weights models measured via hosted inference for speed; all are
locally deployable in principle.

| model | params (B) | family | gated P @ C | union P @ C | validity | kept/dropped | runtime (s) |
|---|---|---|---|---|---|---|---|
| scrubdata-ft-v6 (Qwen3-4B fine-tune) | 4 | qwen3 (fine-tuned) | **0.993** @ 0.287 | 0.905 @ 0.413 | — | 132/38 | — (prior measurement) |
| gpt-oss:20b | 20 | openai/gpt-oss | 1.0 @ 0.000* | 0.845 @ 0.257* | 0.0 | 0/0 | 360 |
| devstral-small-2:24b | 24 | mistral/devstral | 0.943 @ 0.426 | 0.915 @ **0.485** | 1.0 | 208/87 | 135 |
| nemotron-3-nano:30b | 30 | nvidia/nemotron | 1.0 @ 0.138 | 0.877 @ 0.336 | 0.4 | 63/6 | 114 |
| gemma4:31b | 31 | google/gemma | 0.943 @ 0.426 | **0.915 @ 0.485** | 1.0 | 209/28 | 104 |

\* gpt-oss:20b is a serving-path failure, not a measured capability: the model
generated ~4.8k tokens per planning call (`done_reason=stop`) but the Ollama Cloud
proxy returned empty `content` and empty `thinking` on all 5 calls at both
num_predict=4000 and 8000 (simple prompts work) — its "gated" point is the degenerate
empty plan and its "union" point is the heuristic backstop alone. nemotron-3-nano
produced valid JSON on only 2/5 batch calls at num_predict=8000 (long-thinking
truncation); validity is part of the measurement.

**Interpretation.** Zero-shot capability at 24–31B does close — and slightly
exceed — the 4B fine-tune's gap inside the same verifier harness: devstral-24B and
gemma4-31B both land at union 0.915 precision @ 0.485 coverage vs the fine-tune's
0.905 @ 0.413, though the fine-tune remains the most precise gated planner
(0.993 vs 0.943) and the only ≤4B point, while two of the four bigger families
(gpt-oss, nemotron) fail on plan-schema validity before capability even gets
measured. Gemma4-31B is the best family on balance: same gate point as devstral but
cleaner raw plans (verifier dropped 28 entries vs devstral's 87 — vs 38 for the 4B
fine-tune) and the fastest wall-clock (104s). The union still dominates everywhere:
every model's union point adds coverage over its gated point at gate-passing
precision, and it floors even the broken planners (nemotron 0.877 @ 0.336) because
the grounded heuristic covers whatever the model misses.

Artifacts: `eval/results/scaling_arm.json` (rows + provenance),
`eval/results/scaling_<model>_hospital_raw_plan.json` (captured raw plans),
runner: `eval/scaling_arm.py`.
