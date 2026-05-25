import re

from core.tools.calculator import calculate
from core.tools.datetime_tool import get_datetime
from core.tools.registry import ToolRegistry


def test_calculator_allows_safe_math() -> None:
    assert calculate("2 + 2") == "4.0"
    assert "error" in calculate("__import__('os').system('whoami')").lower()


def test_datetime_formats_output() -> None:
    out = get_datetime("UTC")
    assert re.match(r"\d{4}-\d{2}-\d{2}", out)


async def _echo(value: str) -> str:
    return value


async def test_tool_registry_dispatch() -> None:
    registry = ToolRegistry()
    registry.register(
        "echo",
        _echo,
        {"description": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}},
    )
    assert "echo" in registry.list_tools()
    assert await registry.dispatch("echo", {"value": "ok"}) == "ok"
