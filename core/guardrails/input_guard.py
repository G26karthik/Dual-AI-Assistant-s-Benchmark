from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Literal

from huggingface_hub import AsyncInferenceClient

from core.guardrails.patterns import PII_PATTERNS, PROMPT_INJECTION_PATTERNS
from core.guardrails.toxicity import classify_toxicity
from core.observability.langfuse_tracer import get_tracer


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
    inference_token = (
        os.getenv("HF_INFERENCE_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or None
    )
    client = AsyncInferenceClient(model=model, token=inference_token)
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
        self.toxicity_threshold = float(os.getenv("GUARDRAIL_TOXICITY_THRESHOLD", "0.8"))
        self._dangerous_pattern = re.compile(
            r"\b(bomb|bombs|explosive|explosives|molotov|detonator|poison|how to kill|make a weapon|build a gun)\b",
            flags=re.IGNORECASE,
        )

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
        tracer = get_tracer()
        with tracer.span(
            "guard.input.check",
            input={"text_preview": text[:240]},
        ) as span:
            token_len = await asyncio.to_thread(self._token_len, text)
            if token_len > self.max_input_tokens:
                result = GuardResult(allowed=False, reason="Input too long")
                span.update(output=result.__dict__)
                return result

            injection_score = await asyncio.to_thread(self._injection_score, text)
            pii_detected = await asyncio.to_thread(self._pii_detected, text)
            _ = await asyncio.to_thread(self._scrub, text)
            toxicity_score = await classify_toxicity(text)

            if injection_score >= 0.7:
                result = GuardResult(
                    allowed=False,
                    reason="Potential prompt injection detected",
                    pii_detected=pii_detected,
                    injection_score=injection_score,
                    toxicity_score=toxicity_score,
                )
                span.update(output=result.__dict__)
                return result

            if self._dangerous_pattern.search(text):
                result = GuardResult(
                    allowed=False,
                    reason="Disallowed harmful intent detected",
                    pii_detected=pii_detected,
                    injection_score=injection_score,
                    toxicity_score=toxicity_score,
                )
                span.update(output=result.__dict__)
                return result

            if toxicity_score >= self.toxicity_threshold:
                result = GuardResult(
                    allowed=False,
                    reason="Toxicity classifier blocked input",
                    pii_detected=pii_detected,
                    injection_score=injection_score,
                    toxicity_score=toxicity_score,
                    violated_category="toxicity",
                )
                span.update(output=result.__dict__)
                return result

            try:
                llama = await llama_guard_check(text, role="user")
            except Exception as exc:
                result = GuardResult(
                    allowed=True,
                    reason="Llama Guard unavailable; toxicity fallback clear",
                    pii_detected=pii_detected,
                    injection_score=injection_score,
                    toxicity_score=toxicity_score,
                )
                span.record_error(exc)
                span.update(output=result.__dict__)
                return result

            llama.pii_detected = pii_detected
            llama.injection_score = injection_score
            llama.toxicity_score = max(llama.toxicity_score, toxicity_score)
            span.update(output=llama.__dict__)
            return llama
