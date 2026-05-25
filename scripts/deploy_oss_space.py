"""Deploy the OSS assistant to the live HF Space.

Usage:
    python scripts/deploy_oss_space.py [--space USER/SPACE]

Stages ``apps/oss-assistant/*`` plus the shared ``core/`` package into a
temp directory and uploads them with ``HfApi.upload_folder``. ``HF_TOKEN``
must be present in the environment (the project ``.env`` is auto-loaded).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def _stage(repo_root: Path, staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    app_dir = repo_root / "apps" / "oss-assistant"
    for name in ("app.py", "assistant.py", "requirements.txt", "runtime.txt", "README.md"):
        src = app_dir / name
        if src.exists():
            shutil.copy2(src, staging / name)

    core_src = repo_root / "core"
    core_dst = staging / "core"
    shutil.copytree(
        core_src,
        core_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    env_example = repo_root / ".env.example"
    if env_example.exists():
        shutil.copy2(env_example, staging / ".env.example")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space",
        default="LuciferMrng/dual-ai-assistant-benchmark-oss",
        help="HF Space repo id (user/space).",
    )
    parser.add_argument(
        "--staging",
        default=".hf-upload-tmp",
        help="Directory to stage files in before upload.",
    )
    parser.add_argument(
        "--commit-message",
        default="Fix chat-history escaping, force web_search, count blocked turns, render reasoning",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("ERROR: HF_TOKEN is not set.", file=sys.stderr)
        return 1

    staging = repo_root / args.staging
    _stage(repo_root, staging)
    print(f"Staged files to {staging}")

    api = HfApi(token=token)
    info = api.upload_folder(
        folder_path=str(staging),
        repo_id=args.space,
        repo_type="space",
        commit_message=args.commit_message,
        ignore_patterns=["__pycache__", "*.pyc", ".env"],
    )
    print(f"Upload complete. Commit URL: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
