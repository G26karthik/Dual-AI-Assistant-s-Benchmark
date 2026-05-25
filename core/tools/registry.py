from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

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
        try:
            return await fn(**args)
        except Exception as exc:  # pragma: no cover - runtime safety fallback
            return f"Tool '{name}' failed: {exc}"

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())
