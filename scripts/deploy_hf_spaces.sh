#!/bin/bash
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi

SPACE_REPO=${1:-""}
if [ -z "$SPACE_REPO" ]; then
  echo "Usage: ./scripts/deploy_hf_spaces.sh <hf-username/space-name>"
  exit 1
fi

TMP_DIR=$(mktemp -d)
git clone "https://huggingface.co/spaces/${SPACE_REPO}" "$TMP_DIR"
cp -R apps/oss-assistant/* "$TMP_DIR/"
cp -R core "$TMP_DIR/core"
cp .env.example "$TMP_DIR/.env.example"

cd "$TMP_DIR"
git add .
git commit -m "Deploy OSS assistant updates" || true
git push
