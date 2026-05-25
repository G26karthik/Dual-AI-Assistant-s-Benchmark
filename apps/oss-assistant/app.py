from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr
from assistant import OSS_SYSTEM_PROMPT, generate_oss_response
from dotenv import load_dotenv

from core.guardrails.input_guard import InputGuard
from core.guardrails.output_guard import OutputGuard
from core.memory.token_budget import TokenBudgetMemory
from core.observability.logger import StructuredLogger
from core.observability.metrics import MetricsCollector

load_dotenv()

LOGGER = StructuredLogger("oss_assistant")


async def respond(
    message: str,
    history: list[dict[str, Any]],
    enable_web_search: bool,
    show_reasoning: bool,
    memory: TokenBudgetMemory,
    metrics: MetricsCollector,
) -> AsyncGenerator[str, None]:
    _ = history
    session_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    input_guard = InputGuard()
    output_guard = OutputGuard()
    start = time.perf_counter()

    in_result = await input_guard.check(message)
    if not in_result.allowed:
        yield f"Request blocked by input guardrail: {in_result.reason or 'Unknown reason'}"
        return

    memory.add_turn("user", message)
    response, tool_trace = await generate_oss_response(memory, message, enable_web_search=enable_web_search)
    out_result = await output_guard.check(
        message,
        response,
        input_blocked=False,
        used_web_search=any(t["tool"] == "web_search" for t in tool_trace),
        selfcheck_ran=False,
    )
    final_text = response if out_result.allowed else f"Response blocked: {out_result.reason or 'Unsafe output'}"
    memory.add_turn("assistant", final_text)

    words = final_text.split(" ")
    running = ""
    for word in words:
        running = f"{running} {word}".strip()
        yield running
        await asyncio.sleep(0.01)

    latency_ms = int((time.perf_counter() - start) * 1000)
    entry = {
        "session_id": session_id,
        "turn_id": turn_id,
        "model": "qwen2.5-1.5b-instruct",
        "user_input": message,
        "assistant_output": final_text,
        "input_guard": in_result.__dict__,
        "output_guard": out_result.__dict__,
        "tools_used": [t["tool"] for t in tool_trace],
        "tool_calls": tool_trace if show_reasoning else [],
        "latency_ms": {"total": latency_ms},
        "tokens": {
            "prompt_tokens": max(1, len(message) // 4),
            "completion_tokens": max(1, len(final_text) // 4),
            "total_tokens": max(1, (len(message) + len(final_text)) // 4),
        },
        "memory_tokens_in_context": memory.token_count(),
        "memory_tokens_remaining": memory.tokens_remaining(),
        "estimated_cost_usd": 0.0,
    }
    metrics.record_turn(entry)
    await LOGGER.log_turn(entry)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Assistant — OSS") as demo:
        gr.Markdown("## AI Assistant — OSS\nModel: Qwen2.5-0.5B | Status: Ready")
        with gr.Row():
            with gr.Column(scale=3):
                enable_web_search = gr.Checkbox(label="Enable Web Search", value=True)
                show_reasoning = gr.Checkbox(label="Show Reasoning", value=False)
                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        enable_web_search,
                        show_reasoning,
                        gr.State(TokenBudgetMemory(system_prompt=OSS_SYSTEM_PROMPT)),
                        gr.State(MetricsCollector()),
                    ],
                    type="messages",
                )
            with gr.Column(scale=1):
                gr.JSON(label="Session Metrics", value={})
                with gr.Accordion("Guardrail Log", open=False):
                    gr.Markdown("Guardrail events are written to structured logs.")
        _ = chatbot
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_port=args.server_port)


if __name__ == "__main__":
    main()
