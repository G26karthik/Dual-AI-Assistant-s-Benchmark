#!/bin/bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python apps/oss-assistant/app.py --server_port 7860 &
OSS_PID=$!
python apps/frontier-assistant/app.py --server_port 7861 &
FRONTIER_PID=$!

echo "OSS Assistant:      http://localhost:7860"
echo "Frontier Assistant: http://localhost:7861"

wait "$OSS_PID" "$FRONTIER_PID"
