from pathlib import Path

import pytest

from core.observability.logger import StructuredLogger
from core.observability.metrics import MetricsCollector
from core.tools.datetime_tool import get_datetime
from core.tools.web_search import web_search


@pytest.mark.asyncio
async def test_logger_writes_json_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    logger = StructuredLogger("test_logger")
    await logger.log_turn({"session_id": "s1", "message": "ok"})
    logfile = tmp_path / "test_logger.jsonl"
    assert logfile.exists()
    content = logfile.read_text(encoding="utf-8")
    assert '"session_id": "s1"' in content


def test_metrics_summary() -> None:
    collector = MetricsCollector()
    collector.record_turn(
        {
            "session_id": "s1",
            "latency_ms": {"total": 100},
            "tokens": {"total_tokens": 42},
            "estimated_cost_usd": 0.01,
            "input_guard": {"allowed": True},
            "output_guard": {"allowed": True},
            "tool_calls": [{"tool": "calculate"}],
        }
    )
    summary = collector.summary()
    assert summary["avg_latency_ms"] == 100
    assert summary["total_tokens"] == 42
    assert summary["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_web_search_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = await web_search("test query")
    assert "missing TAVILY_API_KEY" in result


@pytest.mark.asyncio
async def test_web_search_handles_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")

    class FailingClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def post(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("boom")

    monkeypatch.setattr("core.tools.web_search.httpx.AsyncClient", FailingClient)
    result = await web_search("test query")
    assert "Web search error" in result


def test_datetime_invalid_timezone() -> None:
    assert "Invalid timezone" in get_datetime("Not/AZone")
