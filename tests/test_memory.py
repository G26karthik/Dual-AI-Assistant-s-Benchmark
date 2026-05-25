import copy

from core.memory.token_budget import TokenBudgetMemory


def test_memory_respects_budget_with_pair_drops() -> None:
    memory = TokenBudgetMemory(budget_tokens=20, system_prompt="sys")
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "hi")
    memory.add_turn("user", "this is a long message that should force eviction")
    memory.add_turn("assistant", "acknowledged")

    messages = memory.get_messages()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "sys"
    assert memory.token_count() <= 20


def test_memory_roundtrip_serialization() -> None:
    memory = TokenBudgetMemory(budget_tokens=100, system_prompt="s")
    memory.add_turn("user", "u1")
    memory.add_turn("assistant", "a1")

    restored = TokenBudgetMemory.from_dict(memory.to_dict())
    assert restored.get_messages() == memory.get_messages()


def test_memory_deepcopy_preserves_state() -> None:
    memory = TokenBudgetMemory(budget_tokens=64, system_prompt="system")
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "world")

    cloned = copy.deepcopy(memory)
    assert cloned.get_messages() == memory.get_messages()
    assert cloned.token_count() == memory.token_count()
