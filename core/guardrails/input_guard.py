from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Literal

from huggingface_hub import AsyncInferenceClient

from core.guardrails.patterns import PII_PATTERNS, PROMPT_INJECTION_PATTERNS


@dataclass
class GuardResult:
    allowed: bool
    reason: str | None = None
    pii_detected: bool = False
    injection_score: float = 0.0
    toxicity_score: float = 0.0
    violated_category: str | None = None


def format_llama_guard_prompt(text: str, role: Literal["user", "agent"]) -> str:
    return f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n{text}\n<|eot_id|>"


def parse_violated_category(response: str) -> str | None:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    return lines[1]


async def llama_guard_check(text: str, role: Literal["user", "agent"]) -> GuardResult:
    model = os.getenv("LLAMA_GUARD_MODEL_ID", "meta-llama/Llama-Guard-3-1B")
    client = AsyncInferenceClient(model=model, token=os.getenv("HF_TOKEN"))
    prompt = format_llama_guard_prompt(text, role)
    response = await client.text_generation(prompt, max_new_tokens=20)
    normalized = response.strip().lower()
    safe = normalized.startswith("safe")
    category = parse_violated_category(response) if not safe else None
    return GuardResult(
        allowed=safe,
        reason=category if not safe else None,
        toxicity_score=0.0 if safe else 1.0,
        violated_category=category,
    )


class InputGuard:
    def __init__(self) -> None:
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "1024"))

    @staticmethod
    def _token_len(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _scrub(text: str) -> str:
        scrubbed = text
        for pattern in PII_PATTERNS.values():
            scrubbed = re.sub(pattern, "[REDACTED]", scrubbed, flags=re.IGNORECASE)
        return scrubbed

    @staticmethod
    def _injection_score(text: str) -> float:
        hits = 0
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits += 1
        if hits == 0:
            return 0.0
        return min(1.0, hits / 3.0)

    @staticmethod
    def _pii_detected(text: str) -> bool:
        for pattern in PII_PATTERNS.values():
            if re.search(pattern, text):
                return True
        return False

    async def check(self, text: str) -> GuardResult:
        token_len = await asyncio.to_thread(self._token_len, text)
        if token_len > self.max_input_tokens:
            return GuardResult(allowed=False, reason="Input too long")

        injection_score = await asyncio.to_thread(self._injection_score, text)
        pii_detected = await asyncio.to_thread(self._pii_detected, text)
        _ = await asyncio.to_thread(self._scrub, text)

        if injection_score >= 0.7:
            return GuardResult(
                allowed=False,
                reason="Potential prompt injection detected",
                pii_detected=pii_detected,
                injection_score=injection_score,
            )

        try:
            llama = await llama_guard_check(text, role="user")
        except Exception:
            return GuardResult(
                allowed=True,
                reason="Llama Guard unavailable; default allow",
                pii_detected=pii_detected,
                injection_score=injection_score,
            )

        llama.pii_detected = pii_detected
        llama.injection_score = injection_score
        return llama
