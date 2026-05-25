"""Behavioural tests for the OSS Gradio app and assistant module.

These cover the four user-visible regressions reported on the deployed Space:

1. ``_normalize_chat_history`` must produce ``{role, content: str}`` even when
   Gradio 6 hands content back as a list of typed parts -- so multi-turn
   history never gets stringified into ``"[{'text': '...'}]"`` chunks that
   then get re-wrapped on the next turn.
2. The web-search heuristic ``_should_use_web_search`` must fire for factual
   / recency questions and stay silent for pure math or bare datetime asks.
3. The ``respond`` handler must record a turn into ``MetricsCollector`` even
   when the input guardrail blocks the prompt, so ``guardrail_blocks``
   actually increments.
4. The ``respond`` handler must always thread ``tool_trace`` into the metrics
   entry so ``tool_call_count`` reflects tool usage regardless of whether
   the user has the ``Show Reasoning`` toggle enabled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "apps" / "oss-assistant"


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, APP_DIR / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assistant_mod():
    return _load_module("oss_assistant_assistant", "assistant.py")


@pytest.fixture(scope="module")
def app_mod(assistant_mod):
    _ = assistant_mod
    return _load_module("oss_assistant_app", "app.py")


def test_extract_text_handles_plain_string(app_mod) -> None:
    assert app_mod._extract_text_from_content("hello") == "hello"


def test_extract_text_handles_gradio_parts_list(app_mod) -> None:
    parts = [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
    assert app_mod._extract_text_from_content(parts) == "hello world"


def test_extract_text_handles_nested_dict(app_mod) -> None:
    nested = {"type": "text", "text": "ok"}
    assert app_mod._extract_text_from_content(nested) == "ok"


def test_normalize_history_roundtrip_is_idempotent(app_mod) -> None:
    """The classic bug: a turn previously written as ``{role, content: 'x'}``
    comes back from Gradio as ``{role, content: [{'type': 'text', 'text': 'x'}]}``
    and must normalise back to the original plain-string form -- not to the
    string ``"[{'type': 'text', 'text': 'x'}]"``.
    """
    history_from_gradio = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi back!"}]},
    ]
    normalized = app_mod._normalize_chat_history(history_from_gradio)
    assert normalized == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi back!"},
    ]

    # Now simulate Gradio re-wrapping each string content on the next turn.
    rewrapped = [
        {"role": item["role"], "content": [{"type": "text", "text": item["content"]}]}
        for item in normalized
    ]
    assert app_mod._normalize_chat_history(rewrapped) == normalized


def test_should_use_web_search_factual_query(assistant_mod) -> None:
    assert assistant_mod._should_use_web_search(
        "what is current leaderboard no 1 in IPL"
    ) is True
    assert assistant_mod._should_use_web_search("Who won the 2025 election?") is True
    assert assistant_mod._should_use_web_search("latest news on AI") is True


def test_should_use_web_search_skips_math(assistant_mod) -> None:
    assert assistant_mod._should_use_web_search("2 + 2") is False
    assert assistant_mod._should_use_web_search("   ") is False


def test_should_use_web_search_skips_bare_datetime(assistant_mod) -> None:
    assert assistant_mod._should_use_web_search("what is the time") is False
    assert assistant_mod._should_use_web_search("date?") is False


@pytest.mark.asyncio
async def test_respond_records_blocked_input_into_metrics(
    app_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocked inputs must still record a turn so ``guardrail_blocks`` ticks."""
    from core.guardrails.input_guard import GuardResult
    from core.memory.token_budget import TokenBudgetMemory
    from core.observability.metrics import MetricsCollector

    async def _block(self: Any, text: str) -> GuardResult:
        _ = (self, text)
        return GuardResult(allowed=False, reason="blocked-by-test")

    async def _noop_log(self: Any, entry: dict[str, Any]) -> None:
        _ = (self, entry)

    monkeypatch.setattr(app_mod.InputGuard, "check", _block, raising=True)
    monkeypatch.setattr(app_mod.LOGGER, "log_turn", _noop_log.__get__(app_mod.LOGGER), raising=True)

    memory = TokenBudgetMemory(system_prompt="sys")
    metrics = MetricsCollector()
    (
        updated_history,
        _,
        metrics_md,
        guardrail_md,
        reasoning_md,
        *_rest,
    ) = await app_mod.respond(
        "anything that gets blocked",
        [],
        False,
        True,
        memory,
        metrics,
        [],
    )

    summary = metrics.summary()
    assert summary["guardrail_blocks"] == 1
    assert summary["sessions_count"] == 1
    assert "blocked-by-test" in guardrail_md
    assert updated_history[-1]["role"] == "assistant"
    assert "blocked" in updated_history[-1]["content"].lower()
    assert "blocked" in metrics_md.lower() or "Guardrail blocks" in metrics_md
    # Reasoning toggle is ON but no tools ran for a blocked prompt -- it
    # should still render a sensible message, not crash.
    assert "No tool calls" in reasoning_md or "Tool Trace" in reasoning_md


@pytest.mark.asyncio
async def test_respond_counts_tool_calls_when_reasoning_off(
    app_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool calls should be counted by metrics regardless of the Show
    Reasoning toggle. Previously ``tool_calls`` was only written into the
    log entry when ``show_reasoning`` was True, so the counter stayed at 0.
    """
    from core.guardrails.input_guard import GuardResult
    from core.guardrails.output_guard import OutputGuard
    from core.memory.token_budget import TokenBudgetMemory
    from core.observability.metrics import MetricsCollector

    async def _allow_input(self: Any, text: str) -> GuardResult:
        _ = (self, text)
        return GuardResult(allowed=True)

    async def _allow_output(self: Any, *args: Any, **kwargs: Any) -> GuardResult:
        _ = (self, args, kwargs)
        return GuardResult(allowed=True)

    async def _fake_generate(memory: Any, message: str, *, enable_web_search: bool):
        _ = (memory, enable_web_search)
        trace = [{"tool": "web_search", "args": {"query": message}, "result": "stub-result"}]
        return "stub answer", trace

    async def _noop_log(self: Any, entry: dict[str, Any]) -> None:
        _ = (self, entry)

    monkeypatch.setattr(app_mod.InputGuard, "check", _allow_input, raising=True)
    monkeypatch.setattr(OutputGuard, "check", _allow_output, raising=True)
    monkeypatch.setattr(app_mod, "generate_oss_response", _fake_generate, raising=True)
    monkeypatch.setattr(app_mod.LOGGER, "log_turn", _noop_log.__get__(app_mod.LOGGER), raising=True)

    memory = TokenBudgetMemory(system_prompt="sys")
    metrics = MetricsCollector()
    await app_mod.respond(
        "factual question",
        [],
        True,
        False,  # show_reasoning OFF
        memory,
        metrics,
        [],
    )
    summary = metrics.summary()
    assert summary["tool_call_count"] == 1
    assert summary["sessions_count"] == 1


@pytest.mark.asyncio
async def test_generate_oss_response_runs_explicit_web_search(
    assistant_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the toggle is on and a query looks factual, ``web_search`` must
    appear in the returned tool trace even if the LLM never emits an
    OpenAI-style ``tool_calls`` block.
    """
    from core.memory.token_budget import TokenBudgetMemory

    async def _fake_web_search(query: str, max_results: int = 3) -> str:
        _ = (query, max_results)
        return "1. Mumbai Indians lead the table.\nURL: https://example/test\nSnippet: stub"

    async def _fake_hf_chat(messages: list[dict[str, str]], tools: list[dict]) -> Any:
        _ = (messages, tools)
        raise RuntimeError("simulate hosted-inference unavailable")

    async def _fake_openrouter(messages: list[dict[str, str]]) -> str:
        _ = messages
        return "Mumbai Indians are #1 on the IPL leaderboard based on the search."

    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    monkeypatch.setattr(assistant_mod, "web_search", _fake_web_search, raising=True)
    monkeypatch.setattr(assistant_mod, "_hf_chat", _fake_hf_chat, raising=True)
    monkeypatch.setattr(assistant_mod, "_openrouter_oss_chat", _fake_openrouter, raising=True)

    memory = TokenBudgetMemory(system_prompt=assistant_mod.OSS_SYSTEM_PROMPT)
    memory.add_turn("user", "what is current leaderboard no 1 in IPL")

    answer, trace = await assistant_mod.generate_oss_response(
        memory,
        "what is current leaderboard no 1 in IPL",
        enable_web_search=True,
    )
    tools_used = [entry["tool"] for entry in trace]
    assert "web_search" in tools_used
    assert "Mumbai" in answer


# --- Prompt-engineering fix for small-model grounding ------------------------


def test_is_search_recall_command_detects_meta_followups(assistant_mod) -> None:
    """``search it`` / ``google this`` / ``look it up`` must route through the
    web-search path even though they contain no question word."""
    assert assistant_mod._is_search_recall_command("search it") is True
    assert assistant_mod._is_search_recall_command("Search it.") is True
    assert assistant_mod._is_search_recall_command("google this") is True
    assert assistant_mod._is_search_recall_command("look it up") is True
    assert assistant_mod._is_search_recall_command("lookup") is True
    assert assistant_mod._is_search_recall_command("please search") is True
    assert assistant_mod._is_search_recall_command("what is the capital of France?") is False
    assert assistant_mod._is_search_recall_command("2 + 2") is False


def test_resolve_search_query_uses_prior_user_turn(assistant_mod) -> None:
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "who won IPL 2025"},
        {"role": "assistant", "content": "I don't know."},
        {"role": "user", "content": "search it"},
    ]
    assert (
        assistant_mod._resolve_search_query("search it", history)
        == "who won IPL 2025"
    )


def test_resolve_search_query_falls_back_to_self(assistant_mod) -> None:
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search it"},
    ]
    assert assistant_mod._resolve_search_query("search it", history) == "search it"


def test_compact_search_grounding_truncates_long_snippets(assistant_mod) -> None:
    long_snippet = "x" * 600
    raw = (
        "1. Headline one\n"
        "URL: https://example.test/a\n"
        f"Snippet: {long_snippet}\n\n"
        "2. Headline two\n"
        "URL: https://example.test/b\n"
        "Snippet: short\n\n"
        "3. Headline three\n"
        "URL: https://example.test/c\n"
        "Snippet: also short\n\n"
        "4. Headline four\n"
        "URL: https://example.test/d\n"
        "Snippet: four\n\n"
        "5. Headline five\n"
        "URL: https://example.test/e\n"
        "Snippet: five\n\n"
        "6. Headline six (should be dropped)\n"
        "URL: https://example.test/f\n"
        "Snippet: six\n"
    )
    compact = assistant_mod._compact_search_grounding(raw)
    assert "Headline one" in compact
    assert "Headline five" in compact
    assert "Headline six" not in compact
    long_line = next(line for line in compact.splitlines() if line.lstrip().startswith("xxx"))
    assert len(long_line) <= 290  # 280-char budget + the two-space indent + ellipsis


def test_inject_grounding_before_user_reorders_messages(assistant_mod) -> None:
    """The grounding block must end up immediately before the (restated) user
    turn so the small OSS model sees facts → question → answer."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": assistant_mod.OSS_SYSTEM_PROMPT},
        {"role": "user", "content": "search it"},
    ]
    raw_grounding = (
        "1. Mumbai Indians win the 2025 IPL final.\n"
        "URL: https://example.test/ipl\n"
        "Snippet: Mumbai Indians defeated Chennai Super Kings in the 2025 IPL final."
    )
    assistant_mod._inject_grounding_before_user(
        messages,
        "search it",
        raw_grounding,
        search_query="who won IPL 2025",
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "system", "user"]
    assert "WEB_SEARCH_RESULTS" in messages[-2]["content"]
    assert "Mumbai Indians" in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert "who won IPL 2025" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_generate_oss_response_handles_search_recall_followup(
    assistant_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare follow-up like ``search it`` must trigger a search against the
    previous real user question instead of being answered from stale memory."""
    from core.memory.token_budget import TokenBudgetMemory

    captured_query: dict[str, str] = {}

    async def _fake_web_search(query: str, max_results: int = 3) -> str:
        captured_query["query"] = query
        return (
            "1. Mumbai Indians win 2025 IPL final.\n"
            "URL: https://example.test/ipl-final\n"
            "Snippet: Mumbai Indians defeated Chennai Super Kings."
        )

    async def _fake_hf_chat(messages: list[dict[str, str]], tools: list[dict]) -> Any:
        _ = (messages, tools)
        raise RuntimeError("simulate hosted-inference unavailable")

    async def _fake_openrouter(messages: list[dict[str, str]]) -> str:
        _ = messages
        return "Mumbai Indians won the 2025 IPL."

    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    monkeypatch.setattr(assistant_mod, "web_search", _fake_web_search, raising=True)
    monkeypatch.setattr(assistant_mod, "_hf_chat", _fake_hf_chat, raising=True)
    monkeypatch.setattr(assistant_mod, "_openrouter_oss_chat", _fake_openrouter, raising=True)

    memory = TokenBudgetMemory(system_prompt=assistant_mod.OSS_SYSTEM_PROMPT)
    memory.add_turn("user", "who won IPL 2025")
    memory.add_turn("assistant", "I do not have current data on that.")
    memory.add_turn("user", "search it")

    answer, trace = await assistant_mod.generate_oss_response(
        memory,
        "search it",
        enable_web_search=True,
    )
    assert captured_query.get("query") == "who won IPL 2025"
    tools_used = [entry["tool"] for entry in trace]
    assert "web_search" in tools_used
    assert "Mumbai" in answer
