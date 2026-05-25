from __future__ import annotations

import asyncio
import os
import re

from core.guardrails.input_guard import GuardResult, llama_guard_check


class OutputGuard:
    def __init__(self) -> None:
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

    @staticmethod
    def _token_len(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _has_high_conf_numeric_claim(text: str) -> bool:
        # Conservative heuristic: many numbers and claim verbs in a long response.
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
        claim_words = ["is", "are", "was", "were", "equals", "approximately"]
        has_claim = any(word in text.lower() for word in claim_words)
        return len(numbers) >= 3 and has_claim

    async def check(
        self,
        prompt: str,
        response: str,
        *,
        input_blocked: bool = False,
        used_web_search: bool = False,
        selfcheck_ran: bool = False,
    ) -> GuardResult:
        try:
            result = await llama_guard_check(response, role="agent")
        except Exception:
            result = GuardResult(allowed=True, reason="Llama Guard unavailable; default allow")

        if input_blocked and result.allowed:
            result.allowed = False
            result.reason = "Input was blocked; output consistency enforcement"

        token_len = await asyncio.to_thread(self._token_len, response)
        if token_len > self.max_output_tokens:
            result.allowed = False
            result.reason = "Output too long"

        if (not selfcheck_ran) and (not used_web_search) and token_len > 100:
            if await asyncio.to_thread(self._has_high_conf_numeric_claim, response):
                result.reason = (result.reason or "") + " [unverified]"

        _ = prompt
        return result
