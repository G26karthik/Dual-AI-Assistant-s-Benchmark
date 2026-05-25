from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from core.memory.token_budget import TokenBudgetMemory


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


async def ask_oss(prompt: str, memory: TokenBudgetMemory | None = None) -> str:
    module = _load_module("oss_assistant_module", _root() / "apps" / "oss-assistant" / "assistant.py")
    if memory is None:
        memory = TokenBudgetMemory(system_prompt=module.OSS_SYSTEM_PROMPT)
    memory.add_turn("user", prompt)
    response, _ = await module.generate_oss_response(memory, prompt, enable_web_search=True)
    memory.add_turn("assistant", response)
    return response


async def ask_frontier(prompt: str, memory: TokenBudgetMemory | None = None) -> str:
    module = _load_module(
        "frontier_assistant_module", _root() / "apps" / "frontier-assistant" / "assistant.py"
    )
    if memory is None:
        memory = TokenBudgetMemory(system_prompt=module.FRONTIER_SYSTEM_PROMPT)
    memory.add_turn("user", prompt)
    response, _ = await module.generate_frontier_response(memory, prompt, enable_web_search=True)
    memory.add_turn("assistant", response)
    return response
