"""Fetch and print the latest logs (build/runtime) for an HF Space."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space",
        default="LuciferMrng/dual-ai-assistant-benchmark-oss",
    )
    parser.add_argument("--kind", choices=["build", "run", "both"], default="both")
    parser.add_argument("--lines", type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    token = os.getenv("HF_TOKEN", "").strip() or None
    api = HfApi(token=token)

    def _print_logs(kind: str) -> None:
        print(f"\n=== {kind.upper()} LOGS (tail {args.lines}) ===")
        try:
            entries = list(api.fetch_logs(repo_id=args.space, repo_type="space", level=kind))
        except Exception as exc:
            print(f"<failed to fetch {kind} logs: {exc}>")
            return
        for entry in entries[-args.lines :]:
            line = getattr(entry, "data", entry)
            ts = getattr(entry, "timestamp", "")
            print(f"{ts} {line}".rstrip())

    if args.kind in {"build", "both"}:
        _print_logs("build")
    if args.kind in {"run", "both"}:
        _print_logs("run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
