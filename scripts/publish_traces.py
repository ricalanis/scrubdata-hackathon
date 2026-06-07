"""Publish captured ScrubData agent traces to a Hugging Face Hub dataset.

Earns the "Sharing is Caring / Open trace" bonus quest. Run after you've exercised
the app (so data/traces/scrubdata-traces.jsonl has records).

    huggingface-cli login                 # one-time (needs HF_TOKEN with write)
    uv run scripts/publish_traces.py      # uploads to build-small-hackathon/scrubdata-traces

The default repo is under the hackathon org; override with --repo for your own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "build-small-hackathon/scrubdata-traces"
DEFAULT_FILE = "data/traces/scrubdata-traces.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    ap.add_argument("--file", default=DEFAULT_FILE, help="local JSONL of traces")
    ap.add_argument("--path-in-repo", default="scrubdata-traces.jsonl")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists() or src.stat().st_size == 0:
        print(f"No traces at {src}. Run the app first to generate traces.", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. `uv add huggingface_hub`.", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Update ScrubData agent traces",
    )
    print(f"Uploaded {src} → https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
