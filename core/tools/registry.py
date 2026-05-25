from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from core.observability.langfuse_tracer import get_tracer

AsyncCallable = Callable[..., Awaitable[str]]
Schema = dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[AsyncCallable, Schema]] = {}

    def register(self, name: str, fn: AsyncCallable, schema: Schema) -> None:
        self._tools[name] = (fn, schema)

    def get_openai_schemas(self) -> list[Schema]:
        schemas: list[Schema] = []
        for name, (_, schema) in self._tools.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return schemas

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Tool '{name}' is not available."
        fn, _ = self._tools[name]
        tracer = get_tracer()
        with tracer.span(
            "tool.dispatch",
            input={"tool_name": name, "args": args},
            metadata={"tool_name": name},
        ) as span:
            try:
                result = await fn(**args)
            except Exception as exc:  # pragma: no cover - runtime safety fallback
                span.record_error(exc)
                failure = f"Tool '{name}' failed: {exc}"
                span.update(output={"result": failure})
                return failure
            span.update(output={"result_preview": result[:500]})
            return result

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())
