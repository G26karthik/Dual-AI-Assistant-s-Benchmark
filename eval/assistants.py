from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from core.guardrails.input_guard import GuardResult, InputGuard
from core.guardrails.output_guard import OutputGuard
from core.memory.token_budget import TokenBudgetMemory
from core.observability.cost import usage_to_cost
from core.observability.langfuse_tracer import get_tracer


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class AssistantCallResult:
    text: str
    tool_trace: list[dict[str, Any]]
    tokens: dict[str, int]
    model_name: str
    provider: str
    cost: dict[str, Any]
    guardrail_events: list[str]
    input_guard: dict[str, Any]
    output_guard: dict[str, Any]


def _guard_to_dict(result: GuardResult) -> dict[str, Any]:
    return {
        "allowed": result.allowed,
        "reason": result.reason,
        "pii_detected": result.pii_detected,
        "injection_score": result.injection_score,
        "toxicity_score": result.toxicity_score,
        "violated_category": result.violated_category,
    }


async def _run_assistant_turn(
    *,
    prompt: str,
    memory: TokenBudgetMemory,
    generate_fn,
    provider: str,
    model_name: str,
) -> AssistantCallResult:
    tracer = get_tracer()
    input_guard = InputGuard()
    output_guard = OutputGuard()
    guardrail_events: list[str] = []

    with tracer.span(
        "eval.assistant.turn",
        input={"provider": provider, "model_name": model_name, "prompt_preview": prompt[:200]},
    ) as span:
        input_result = await input_guard.check(prompt)
        if not input_result.allowed:
            blocked_text = f"Request blocked by input guardrail: {input_result.reason or 'Unknown reason'}"
            cost = usage_to_cost(
                model_name=model_name,
                provider=provider,
                usage=None,
                prompt_fallback=prompt,
                completion_fallback=blocked_text,
            )
            guardrail_events.append(f"input: {input_result.reason or 'blocked'}")
            result = AssistantCallResult(
                text=blocked_text,
                tool_trace=[],
                tokens={
                    "prompt_tokens": cost.prompt_tokens,
                    "completion_tokens": cost.completion_tokens,
                    "total_tokens": cost.total_tokens,
                },
                model_name=model_name,
                provider=provider,
                cost=cost.to_dict(),
                guardrail_events=guardrail_events,
                input_guard=_guard_to_dict(input_result),
                output_guard={"allowed": True, "reason": "skipped: input blocked"},
            )
            span.update(output={"blocked": True, "reason": input_result.reason})
            return result

        memory.add_turn("user", prompt)
        response_text, tool_trace = await generate_fn(memory, prompt, enable_web_search=True)
        output_result = await output_guard.check(
            prompt,
            response_text,
            input_blocked=False,
            used_web_search=any(trace.get("tool") == "web_search" for trace in tool_trace),
            selfcheck_ran=False,
        )
        final_text = (
            response_text
            if output_result.allowed
            else f"Response blocked: {output_result.reason or 'Unsafe output'}"
        )
        memory.add_turn("assistant", final_text)
        if not output_result.allowed:
            guardrail_events.append(f"output: {output_result.reason or 'blocked'}")

        cost = usage_to_cost(
            model_name=model_name,
            provider=provider,
            usage=None,
            prompt_fallback=prompt,
            completion_fallback=final_text,
        )
        result = AssistantCallResult(
            text=final_text,
            tool_trace=tool_trace,
            tokens={
                "prompt_tokens": cost.prompt_tokens,
                "completion_tokens": cost.completion_tokens,
                "total_tokens": cost.total_tokens,
            },
            model_name=model_name,
            provider=provider,
            cost=cost.to_dict(),
            guardrail_events=guardrail_events,
            input_guard=_guard_to_dict(input_result),
            output_guard=_guard_to_dict(output_result),
        )
        span.update(output={"tool_count": len(tool_trace), "cost": result.cost})
        return result


async def ask_oss_with_meta(prompt: str, memory: TokenBudgetMemory | None = None) -> AssistantCallResult:
    module = _load_module("oss_assistant_module", _root() / "apps" / "oss-assistant" / "assistant.py")
    if memory is None:
        memory = TokenBudgetMemory(system_prompt=module.OSS_SYSTEM_PROMPT)
    provider = "openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else "huggingface"
    model_name = (
        os.getenv("OSS_OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
        if provider == "openrouter"
        else os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    return await _run_assistant_turn(
        prompt=prompt,
        memory=memory,
        generate_fn=module.generate_oss_response,
        provider=provider,
        model_name=model_name,
    )


async def ask_oss(prompt: str, memory: TokenBudgetMemory | None = None) -> str:
    result = await ask_oss_with_meta(prompt, memory=memory)
    return result.text


async def ask_frontier_with_meta(
    prompt: str,
    memory: TokenBudgetMemory | None = None,
) -> AssistantCallResult:
    module = _load_module(
        "frontier_assistant_module", _root() / "apps" / "frontier-assistant" / "assistant.py"
    )
    if memory is None:
        memory = TokenBudgetMemory(system_prompt=module.FRONTIER_SYSTEM_PROMPT)
    return await _run_assistant_turn(
        prompt=prompt,
        memory=memory,
        generate_fn=module.generate_frontier_response,
        provider="openrouter",
        model_name=os.getenv("OPENROUTER_MODEL", "~openai/gpt-mini-latest"),
    )


async def ask_frontier(prompt: str, memory: TokenBudgetMemory | None = None) -> str:
    result = await ask_frontier_with_meta(prompt, memory=memory)
    return result.text
