"""WS4 baseline: Baran (Mahdavi & Abedjan, VLDB 2020) on the Raha real-error slice.

Runs Baran in its OWN reference configuration (the package's __main__ example):
oracle error positions from the dirty/gold diff + LABELING_BUDGET gold-labeled tuples
(auto-sampled), no Wikipedia-pretrained value models. This is an UPPER BOUND under a
strictly more informed protocol than ours (we are zero-label, no oracle detection) —
disclosed in the paper. With oracle detection Baran only edits true-error cells, so its
damage rate is NEAR-zero structurally — but not exactly 0: raha normalizes values at
CSV load (html-unescape + whitespace collapse), so its repaired output differs from the
raw-loaded dirty table at cells it never corrected (measured churn-neutral damage:
hospital 0.004, rayyan 0.010 — see eval/cross_scoring.py).

STANDALONE on purpose: stdlib + pandas + raha only — it runs inside a pinned ephemeral
env (raha 1.26 is 2023 code), never importing scrubdata:

    uv run --python 3.10 --with "raha==1.26" --with "numpy<2" --with "pandas<2.1" \
      --with "scikit-learn<1.4" python eval/run_baran.py

Outputs eval/results/baran/<name>_seed<k>_repaired.csv; scored in the main env by
eval/baselines_learned.py under the identical churn-neutral protocol.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

DATASETS = ["hospital", "beers", "flights", "rayyan", "movies_1"]
RAW = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets"
BASE = Path(__file__).resolve().parent.parent


def _fetch(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same fetch + movies_1 truncation as eval.run_real_multi._raha_pair (duplicated
    so this file never imports scrubdata inside the pinned env)."""
    d = BASE / "data" / "real" / name
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for fn in ("dirty.csv", "clean.csv"):
        p = d / fn
        if not p.exists():
            urllib.request.urlretrieve(f"{RAW}/{name}/{fn}", p)
        out.append(pd.read_csv(p, dtype=str, keep_default_na=False))
    dirty, clean = out
    if len(dirty) > 2200:
        dirty, clean = dirty.head(2000).reset_index(drop=True), clean.head(2000).reset_index(drop=True)
    return dirty, clean


def baran_repair(dirty_csv: str, clean_csv: str, name: str,
                 n_labels: int = 20, seed: int = 0) -> pd.DataFrame:
    """Official Baran reference config; returns the repaired DataFrame."""
    import numpy as np
    import raha
    random.seed(seed)
    np.random.seed(seed)
    data = raha.dataset.Dataset({"name": name, "path": dirty_csv, "clean_path": clean_csv})
    data.detected_cells = dict(data.get_actual_errors_dictionary())   # oracle detection
    app = raha.correction.Correction()
    app.LABELING_BUDGET = n_labels
    app.SAVE_RESULTS = False
    app.VERBOSE = False
    corrections = app.run(data)                       # {(row, col_idx): value}
    out = data.dataframe.copy()
    for (i, j), v in corrections.items():
        out.iat[i, j] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/baran")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-labels", type=int, default=20)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in args.datasets.split(","):
        dirty, clean = _fetch(name)
        with tempfile.TemporaryDirectory() as td:     # Dataset wants file paths
            dp, cp = os.path.join(td, "dirty.csv"), os.path.join(td, "clean.csv")
            dirty.to_csv(dp, index=False)
            clean.to_csv(cp, index=False)
            for seed in seeds:
                dest = out_dir / f"{name}_seed{seed}_repaired.csv"
                if dest.exists():
                    print(f"skip {dest} (exists)", flush=True)
                    continue
                print(f"baran: {name} seed={seed} ...", flush=True)
                repaired = baran_repair(dp, cp, name, n_labels=args.n_labels, seed=seed)
                repaired.to_csv(dest, index=False)
                print(f"  -> {dest}", flush=True)
    print("baran runs complete")


if __name__ == "__main__":
    main()
