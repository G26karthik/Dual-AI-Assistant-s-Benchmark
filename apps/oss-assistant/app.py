from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import gradio as gr
from assistant import OSS_SYSTEM_PROMPT, generate_oss_response
from dotenv import load_dotenv

from core.guardrails.input_guard import InputGuard
from core.guardrails.output_guard import OutputGuard
from core.memory.token_budget import TokenBudgetMemory
from core.observability.cost import usage_to_cost
from core.observability.langfuse_tracer import get_tracer
from core.observability.logger import StructuredLogger
from core.observability.metrics import MetricsCollector

load_dotenv()

LOGGER = StructuredLogger("oss_assistant")


def _format_metrics(metrics: MetricsCollector) -> str:
    summary = metrics.summary()
    return (
        f"- **Avg latency:** {float(summary.get('avg_latency_ms', 0.0)):.2f} ms\n"
        f"- **P50 / P95 latency:** {float(summary.get('p50_latency_ms', 0.0)):.2f} / "
        f"{float(summary.get('p95_latency_ms', 0.0)):.2f} ms\n"
        f"- **Total tokens:** {int(summary.get('total_tokens', 0))}\n"
        f"- **Actual cost:** ${float(summary.get('actual_cost_usd', 0.0)):.6f}\n"
        f"- **Equivalent cost:** ${float(summary.get('equivalent_cost_usd', 0.0)):.6f}\n"
        f"- **Guardrail blocks:** {int(summary.get('guardrail_blocks', 0))}\n"
        f"- **Tool calls:** {int(summary.get('tool_call_count', 0))}\n"
        f"- **Sessions tracked:** {int(summary.get('sessions_count', 0))}"
    )


def _append_guardrail_event(events: list[str], phase: str, reason: str | None, allowed: bool) -> list[str]:
    stamp = datetime.now(UTC).strftime("%H:%M:%S UTC")
    status = "ALLOWED" if allowed else "BLOCKED"
    detail = reason or "No reason provided"
    events.append(f"- `{stamp}` [{phase}] **{status}** — {detail}")
    return events[-20:]


def _render_guardrail_markdown(events: list[str]) -> str:
    if not events:
        return "No guardrail events yet."
    return "\n".join(events)


def _render_reasoning(show_reasoning: bool, tool_trace: list[dict[str, Any]]) -> str:
    if not show_reasoning:
        return "Reasoning is hidden. Enable **Show Reasoning** to inspect tool traces."
    if not tool_trace:
        return "No tool calls were made for this response."
    lines = ["### Tool Trace"]
    for call in tool_trace:
        tool = str(call.get("tool", "unknown"))
        args = call.get("args", {})
        result = str(call.get("result", ""))
        excerpt = result if len(result) <= 400 else result[:400] + "…"
        lines.append(f"- **`{tool}`** args=`{args}`")
        lines.append(f"  - result: {excerpt}")
    return "\n".join(lines)


def _extract_text_from_content(content: Any) -> str:
    """Coerce a Gradio Chatbot ``content`` field into plain text.

    Gradio 6's ``Chatbot`` (with the default ``type='messages'``) normalises
    every message ``content`` into a list of typed parts such as
    ``[{'type': 'text', 'text': 'hi'}]``. Older versions return plain strings,
    and on the way back into our handler the same field may arrive as either
    shape. Stringifying the list directly is what caused the nested
    ``[{'text': '[{\'text\': ...`` chains visible in the deployed Space, because
    on every turn the previously-stringified list was wrapped again.

    We always return a plain ``str`` here so the value we hand back to the
    Chatbot is the same shape the Chatbot itself expects (a string), making
    the round-trip idempotent regardless of how many turns accumulate.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                inner = part.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                    continue
                if inner is not None:
                    parts.append(_extract_text_from_content(inner))
                    continue
                # Unknown part shape (file/component); skip silently.
                continue
            if isinstance(part, str):
                parts.append(part)
                continue
            parts.append(str(part))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
        if inner is not None:
            return _extract_text_from_content(inner)
        return ""
    return str(content)


def _normalize_chat_history(history: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(history, list):
        return normalized

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            if role is None:
                continue
            normalized.append({
                "role": str(role),
                "content": _extract_text_from_content(item.get("content")),
            })
            continue

        role = getattr(item, "role", None)
        if role is not None:
            normalized.append({
                "role": str(role),
                "content": _extract_text_from_content(getattr(item, "content", None)),
            })
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            normalized.append({"role": "user", "content": _extract_text_from_content(item[0])})
            normalized.append({"role": "assistant", "content": _extract_text_from_content(item[1])})

    return normalized


def _append_chat_turn(history: list[dict[str, str]], user_text: str, assistant_text: str) -> list[dict[str, str]]:
    return [
        *history,
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _runtime_model_info() -> tuple[str, str]:
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "openrouter", os.getenv(
            "OSS_OPENROUTER_MODEL",
            "meta-llama/llama-3.2-3b-instruct:free",
        )
    return "huggingface", os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


def _build_log_entry(
    *,
    session_id: str,
    turn_id: str,
    user_message: str,
    assistant_text: str,
    input_guard_result: Any,
    output_guard_result: Any,
    tool_trace: list[dict[str, Any]],
    latency_ms: int,
    memory: TokenBudgetMemory,
) -> dict[str, Any]:
    provider, model_name = _runtime_model_info()
    cost = usage_to_cost(
        model_name=model_name,
        provider=provider,
        usage=None,
        prompt_fallback=user_message,
        completion_fallback=assistant_text,
    )
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "model": model_name,
        "provider": provider,
        "user_input": user_message,
        "assistant_output": assistant_text,
        "input_guard": input_guard_result,
        "output_guard": output_guard_result,
        "tools_used": [t["tool"] for t in tool_trace],
        "tool_calls": tool_trace,
        "latency_ms": {"total": latency_ms},
        "tokens": {
            "prompt_tokens": _approx_tokens(user_message),
            "completion_tokens": _approx_tokens(assistant_text),
            "total_tokens": _approx_tokens(user_message) + _approx_tokens(assistant_text),
        },
        "memory_tokens_in_context": memory.token_count(),
        "memory_tokens_remaining": memory.tokens_remaining(),
        "cost": cost.to_dict(),
        "estimated_cost_usd": cost.actual_cost_usd,
    }


async def respond(
    message,
    history,
    enable_web_search,
    show_reasoning,
    memory,
    metrics,
    guardrail_events,
):
    tracer = get_tracer()
    normalized_history = _normalize_chat_history(history)
    user_message = _extract_text_from_content(message)
    session_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    input_guard = InputGuard()
    output_guard = OutputGuard()
    start = time.perf_counter()

    with tracer.span(
        "app.oss.respond",
        input={"session_id": session_id, "user_message": user_message[:240]},
    ) as span:
        in_result = await input_guard.check(user_message)
        if not in_result.allowed:
            guardrail_events = _append_guardrail_event(
                guardrail_events,
                phase="input",
                reason=in_result.reason,
                allowed=False,
            )
            bot_reply = f"Request blocked by input guardrail: {in_result.reason or 'Unknown reason'}"
            updated_history = _append_chat_turn(normalized_history, user_message, bot_reply)

            latency_ms = int((time.perf_counter() - start) * 1000)
            blocked_entry = _build_log_entry(
                session_id=session_id,
                turn_id=turn_id,
                user_message=user_message,
                assistant_text=bot_reply,
                input_guard_result=in_result.__dict__,
                output_guard_result={"allowed": True, "reason": "skipped: input blocked"},
                tool_trace=[],
                latency_ms=latency_ms,
                memory=memory,
            )
            metrics.record_turn(blocked_entry)
            await LOGGER.log_turn(blocked_entry)
            span.update(output={"blocked": True, "reason": in_result.reason})

            return (
                updated_history,
                "",
                _format_metrics(metrics),
                _render_guardrail_markdown(guardrail_events),
                _render_reasoning(show_reasoning, []),
                memory,
                metrics,
                guardrail_events,
            )

        memory.add_turn("user", user_message)
        response, tool_trace = await generate_oss_response(
            memory, user_message, enable_web_search=enable_web_search
        )
        out_result = await output_guard.check(
            user_message,
            response,
            input_blocked=False,
            used_web_search=any(t.get("tool") == "web_search" for t in tool_trace),
            selfcheck_ran=False,
        )
        final_text = response if out_result.allowed else f"Response blocked: {out_result.reason or 'Unsafe output'}"
        memory.add_turn("assistant", final_text)
        guardrail_events = _append_guardrail_event(
            guardrail_events,
            phase="output",
            reason=out_result.reason,
            allowed=out_result.allowed,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        entry = _build_log_entry(
            session_id=session_id,
            turn_id=turn_id,
            user_message=user_message,
            assistant_text=final_text,
            input_guard_result=in_result.__dict__,
            output_guard_result=out_result.__dict__,
            tool_trace=tool_trace,
            latency_ms=latency_ms,
            memory=memory,
        )
        metrics.record_turn(entry)
        await LOGGER.log_turn(entry)
        span.update(output={"tool_count": len(tool_trace), "latency_ms": latency_ms})

        updated_history = _append_chat_turn(normalized_history, user_message, final_text)
        return (
            updated_history,
            "",
            _format_metrics(metrics),
            _render_guardrail_markdown(guardrail_events),
            _render_reasoning(show_reasoning, tool_trace),
            memory,
            metrics,
            guardrail_events,
        )


async def respond_api(
    message: str,
    enable_web_search: bool,
    show_reasoning: bool,
) -> dict[str, Any]:
    memory = TokenBudgetMemory(system_prompt=OSS_SYSTEM_PROMPT)
    metrics = MetricsCollector()
    guardrail_events: list[str] = []
    (
        updated_history,
        _,
        metrics_md,
        guardrail_md,
        reasoning_md,
        _memory_state,
        _metrics_state,
        _guardrail_state,
    ) = await respond(
        message,
        [],
        enable_web_search,
        show_reasoning,
        memory,
        metrics,
        guardrail_events,
    )
    assistant_message = ""
    for item in reversed(updated_history):
        if item.get("role") == "assistant":
            assistant_message = str(item.get("content", ""))
            break
    return {
        "response": assistant_message,
        "metrics_markdown": metrics_md,
        "guardrail_markdown": guardrail_md,
        "reasoning_markdown": reasoning_md,
    }


def _theme() -> gr.themes.Soft:
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Assistant — OSS") as demo:
        gr.Markdown(
            "## OSS Personal Assistant\n"
            "**Model:** Qwen2.5-0.5B-Instruct  \n"
            "**Status:** Ready"
        )
        with gr.Row():
            enable_web_search = gr.Checkbox(label="Enable Web Search", value=True)
            show_reasoning = gr.Checkbox(label="Show Reasoning", value=False)
        api_message = gr.Textbox(visible=False)
        api_enable_web_search = gr.Checkbox(visible=False, value=False)
        api_show_reasoning = gr.Checkbox(visible=False, value=False)
        api_response = gr.JSON(visible=False)
        api_call_btn = gr.Button(visible=False)

        with gr.Row():
            with gr.Column(scale=3):
                # Gradio 6 dropped the `type=` argument: the messages format
                # (list of {role, content} dicts) is now the only supported
                # value. The chat-history normalisation logic in
                # ``_normalize_chat_history`` already handles the typed-parts
                # shape Gradio hands back when this list is preprocessed.
                chatbot = gr.Chatbot(label="Chat", height=420)
                user_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask anything...",
                    lines=3,
                )
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Session")

            with gr.Column(scale=2):
                metrics_md = gr.Markdown("No turns yet.")
                with gr.Accordion("Guardrail Log", open=True):
                    guardrail_md = gr.Markdown("No guardrail events yet.")
                with gr.Accordion("Reasoning / Tool Trace", open=False):
                    reasoning_md = gr.Markdown(
                        "Reasoning is hidden. Enable **Show Reasoning** to inspect tool traces."
                    )

        memory_state = gr.State(TokenBudgetMemory(system_prompt=OSS_SYSTEM_PROMPT))
        metrics_state = gr.State(MetricsCollector())
        guardrail_state = gr.State([])

        send_btn.click(
            fn=respond,
            inputs=[
                user_input,
                chatbot,
                enable_web_search,
                show_reasoning,
                memory_state,
                metrics_state,
                guardrail_state,
            ],
            outputs=[
                chatbot,
                user_input,
                metrics_md,
                guardrail_md,
                reasoning_md,
                memory_state,
                metrics_state,
                guardrail_state,
            ],
            api_name=False,
        )

        user_input.submit(
            fn=respond,
            inputs=[
                user_input,
                chatbot,
                enable_web_search,
                show_reasoning,
                memory_state,
                metrics_state,
                guardrail_state,
            ],
            outputs=[
                chatbot,
                user_input,
                metrics_md,
                guardrail_md,
                reasoning_md,
                memory_state,
                metrics_state,
                guardrail_state,
            ],
            api_name=False,
        )

        def clear_session():
            fresh_memory = TokenBudgetMemory(system_prompt=OSS_SYSTEM_PROMPT)
            fresh_metrics = MetricsCollector()
            return (
                [],
                "",
                "No turns yet.",
                "No guardrail events yet.",
                "Reasoning is hidden. Enable **Show Reasoning** to inspect tool traces.",
                fresh_memory,
                fresh_metrics,
                [],
            )

        clear_btn.click(
            fn=clear_session,
            inputs=[],
            outputs=[
                chatbot,
                user_input,
                metrics_md,
                guardrail_md,
                reasoning_md,
                memory_state,
                metrics_state,
                guardrail_state,
            ],
            api_name=False,
        )
        api_call_btn.click(
            fn=respond_api,
            inputs=[api_message, api_enable_web_search, api_show_reasoning],
            outputs=[api_response],
            api_name="respond_api",
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()
    # Gradio 6 moved several Blocks kwargs (including ``theme``) onto
    # ``launch``; pass it here to silence the deprecation warning.
    build_app().launch(server_port=args.server_port, theme=_theme())


if __name__ == "__main__":
    main()
