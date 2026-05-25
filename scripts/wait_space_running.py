"""Poll an HF Space until it reaches RUNNING (or fails)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space",
        default="LuciferMrng/dual-ai-assistant-benchmark-oss",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--interval", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    token = os.getenv("HF_TOKEN", "").strip() or None
    api = HfApi(token=token)

    start = time.time()
    last_stage = ""
    while time.time() - start < args.timeout:
        try:
            runtime = api.get_space_runtime(repo_id=args.space)
        except Exception as exc:
            print(f"[{int(time.time() - start):4d}s] error: {exc}")
            time.sleep(args.interval)
            continue
        stage = getattr(runtime, "stage", str(runtime))
        if stage != last_stage:
            print(f"[{int(time.time() - start):4d}s] stage={stage}")
            last_stage = stage
        if stage == "RUNNING":
            return 0
        if stage in {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "DELETED"}:
            print("Space entered a failure stage.", file=sys.stderr)
            return 2
        time.sleep(args.interval)
    print("Timed out waiting for Space to run.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
