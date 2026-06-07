"""Push the verified SFT data to an HF dataset repo so Colab/HF-Jobs can pull it.

    uv run training/build_dataset.py --n 2000 --out data/train.jsonl   # (re)generate
    huggingface-cli login                                              # HF write token
    uv run scripts/push_dataset.py --repo build-small-hackathon/scrubdata-sft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "build-small-hackathon/scrubdata-sft"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--file", default="data/train.jsonl")
    ap.add_argument("--path-in-repo", default="train.jsonl")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists() or src.stat().st_size == 0:
        print(f"No data at {src}. Run training/build_dataset.py first.", file=sys.stderr)
        return 1
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed.", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=args.path_in_repo,
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Update ScrubData SFT data")
    print(f"Pushed {src} → https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
