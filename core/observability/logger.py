from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, name: str = "assistant") -> None:
        self.log_dir = Path(os.getenv("LOG_DIR", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        if not self.logger.handlers:
            file_handler = RotatingFileHandler(
                self.log_dir / f"{name}.jsonl",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            stream_handler = logging.StreamHandler()
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    async def log_turn(self, entry: dict[str, Any]) -> None:
        payload = dict(entry)
        payload.setdefault("timestamp", datetime.now(UTC).isoformat())
        line = json.dumps(payload, ensure_ascii=True)
        await asyncio.to_thread(self.logger.info, line)
