from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CostEstimate:
    provider: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_cost_usd: float
    equivalent_cost_usd: float
    is_free_tier: bool
    pricing_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_REFERENCE_INPUT_PER_1M = 0.15
_DEFAULT_REFERENCE_OUTPUT_PER_1M = 0.60

_ACTUAL_RATES_PER_1M: dict[tuple[str, str], tuple[float, float]] = {
    ("openrouter", "~openai/gpt-mini-latest"): (0.15, 0.60),
    ("openrouter", "openai/gpt-4.1-mini"): (0.40, 1.60),
    ("openrouter", "openai/gpt-oss-120b:free"): (0.0, 0.0),
    ("openrouter", "meta-llama/llama-3.2-3b-instruct:free"): (0.0, 0.0),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"): (0.0, 0.0),
    ("openrouter", "qwen/qwen3-coder:free"): (0.0, 0.0),
    ("openrouter", "google/gemma-4-31b-it:free"): (0.0, 0.0),
    ("google", "gemini-2.0-flash"): (0.0, 0.0),
}

_EQUIVALENT_RATES_PER_1M: dict[tuple[str, str], tuple[float, float]] = {
    ("google", "gemini-2.0-flash"): (0.15, 0.60),
    ("openrouter", "meta-llama/llama-3.2-3b-instruct:free"): (0.15, 0.60),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"): (0.15, 0.60),
    ("openrouter", "qwen/qwen3-coder:free"): (0.15, 0.60),
    ("openrouter", "google/gemma-4-31b-it:free"): (0.15, 0.60),
    ("openrouter", "openai/gpt-oss-120b:free"): (0.15, 0.60),
}


def _clean_token_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _env_rate(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return max(0.0, parsed)


def approximate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _fallback_reference_rates() -> tuple[float, float]:
    return (
        _env_rate(
            "REFERENCE_INPUT_COST_USD_PER_1M",
            _DEFAULT_REFERENCE_INPUT_PER_1M,
        ),
        _env_rate(
            "REFERENCE_OUTPUT_COST_USD_PER_1M",
            _DEFAULT_REFERENCE_OUTPUT_PER_1M,
        ),
    )


def _lookup_rates(
    provider: str,
    model_name: str,
    table: dict[tuple[str, str], tuple[float, float]],
) -> tuple[float, float] | None:
    exact = table.get((provider, model_name))
    if exact is not None:
        return exact
    stripped = model_name.replace(":free", "")
    return table.get((provider, stripped))


def _is_free_tier(provider: str, model_name: str) -> bool:
    normalized = model_name.lower()
    return (
        ":free" in normalized
        or provider == "huggingface"
        or (provider == "google" and "gemini" in normalized)
    )


def estimate_cost(
    *,
    model_name: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostEstimate:
    clean_prompt_tokens = _clean_token_count(prompt_tokens)
    clean_completion_tokens = _clean_token_count(completion_tokens)
    total_tokens = clean_prompt_tokens + clean_completion_tokens
    normalized_provider = provider.strip().lower() or "unknown"
    normalized_model = model_name.strip() or "unknown"

    actual_rates = _lookup_rates(normalized_provider, normalized_model, _ACTUAL_RATES_PER_1M)
    equivalent_rates = _lookup_rates(
        normalized_provider,
        normalized_model,
        _EQUIVALENT_RATES_PER_1M,
    )
    pricing_source = "model_table" if actual_rates is not None else "reference_fallback"
    if actual_rates is None:
        actual_rates = _fallback_reference_rates()
    if equivalent_rates is None:
        equivalent_rates = _fallback_reference_rates()

    free_tier = _is_free_tier(normalized_provider, normalized_model)
    if free_tier:
        actual_cost_usd = 0.0
    else:
        actual_cost_usd = (
            (clean_prompt_tokens / 1_000_000.0) * actual_rates[0]
            + (clean_completion_tokens / 1_000_000.0) * actual_rates[1]
        )

    equivalent_cost_usd = (
        (clean_prompt_tokens / 1_000_000.0) * equivalent_rates[0]
        + (clean_completion_tokens / 1_000_000.0) * equivalent_rates[1]
    )
    if (not free_tier) and equivalent_cost_usd < actual_cost_usd:
        equivalent_cost_usd = actual_cost_usd

    return CostEstimate(
        provider=normalized_provider,
        model_name=normalized_model,
        prompt_tokens=clean_prompt_tokens,
        completion_tokens=clean_completion_tokens,
        total_tokens=total_tokens,
        actual_cost_usd=round(actual_cost_usd, 8),
        equivalent_cost_usd=round(equivalent_cost_usd, 8),
        is_free_tier=free_tier,
        pricing_source=pricing_source,
    )


def usage_to_cost(
    *,
    model_name: str,
    provider: str,
    usage: Any,
    prompt_fallback: str = "",
    completion_fallback: str = "",
) -> CostEstimate:
    prompt_tokens = 0
    completion_tokens = 0
    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = _clean_token_count(
                usage.get("prompt_tokens") or usage.get("promptTokenCount")
            )
            completion_tokens = _clean_token_count(
                usage.get("completion_tokens")
                or usage.get("candidatesTokenCount")
                or usage.get("output_tokens")
            )
        else:
            prompt_tokens = _clean_token_count(
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "promptTokenCount", None)
            )
            completion_tokens = _clean_token_count(
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "candidatesTokenCount", None)
                or getattr(usage, "output_tokens", None)
            )

    if prompt_tokens == 0 and prompt_fallback:
        prompt_tokens = approximate_tokens(prompt_fallback)
    if completion_tokens == 0 and completion_fallback:
        completion_tokens = approximate_tokens(completion_fallback)

    return estimate_cost(
        model_name=model_name,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
