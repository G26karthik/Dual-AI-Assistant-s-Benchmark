from __future__ import annotations

from threading import Lock
from typing import Any, Literal

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback for minimal envs
    tiktoken = None  # type: ignore[assignment]


MessageRole = Literal["user", "assistant", "system"]


class TokenBudgetMemory:
    """
    Maintains conversation history within a fixed token budget.

    The system prompt is preserved at index 0 and is not counted against the
    rolling history budget.
    """

    def __init__(
        self,
        budget_tokens: int = 4096,
        system_prompt: str = "",
        model: str = "cl100k_base",
    ) -> None:
        self.budget_tokens = budget_tokens
        self.system_prompt = system_prompt
        self.model = model
        self._history: list[dict[str, str]] = []
        self._lock = Lock()
        self._encoder = self._build_encoder(model)

    def _build_encoder(self, model: str) -> Any | None:
        if tiktoken is None:
            return None
        try:
            return tiktoken.get_encoding(model)
        except Exception:
            return None

    def __getstate__(self) -> dict[str, Any]:
        # Gradio deepcopy/pickle support: drop lock/encoder and restore on load.
        return {
            "budget_tokens": self.budget_tokens,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "history": list(self._history),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.budget_tokens = int(state.get("budget_tokens", 4096))
        self.system_prompt = str(state.get("system_prompt", ""))
        self.model = str(state.get("model", "cl100k_base"))
        history = state.get("history", [])
        self._history = []
        if isinstance(history, list):
            for item in history:
                if isinstance(item, dict) and "role" in item and "content" in item:
                    role = str(item["role"])
                    if role in {"user", "assistant"}:
                        self._history.append({"role": role, "content": str(item["content"])})
        self._lock = Lock()
        self._encoder = self._build_encoder(self.model)
        self._enforce_budget()

    def _count_text_tokens(self, text: str) -> int:
        if self._encoder is None:
            # Heuristic fallback for environments without tiktoken or unknown encodings.
            return max(1, len(text) // 4)
        return len(self._encoder.encode(text))

    def _count_message_tokens(self, message: dict[str, str]) -> int:
        return self._count_text_tokens(message["content"]) + self._count_text_tokens(message["role"])

    def _history_token_count(self) -> int:
        return sum(self._count_message_tokens(msg) for msg in self._history)

    def _enforce_budget(self) -> None:
        while self._history and self._history_token_count() > self.budget_tokens:
            # Drop oldest user+assistant pair when possible.
            first_role = self._history[0]["role"]
            self._history.pop(0)
            if first_role == "user" and self._history and self._history[0]["role"] == "assistant":
                self._history.pop(0)

    def add_turn(self, role: Literal["user", "assistant"], content: str) -> None:
        with self._lock:
            self._history.append({"role": role, "content": content})
            self._enforce_budget()

    def get_messages(self) -> list[dict[str, str]]:
        with self._lock:
            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(self._history)
            return messages

    def token_count(self) -> int:
        with self._lock:
            return self._history_token_count()

    def tokens_remaining(self) -> int:
        return max(0, self.budget_tokens - self.token_count())

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_tokens": self.budget_tokens,
                "system_prompt": self.system_prompt,
                "model": self.model,
                "history": list(self._history),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenBudgetMemory:
        memory = cls(
            budget_tokens=int(data.get("budget_tokens", 4096)),
            system_prompt=str(data.get("system_prompt", "")),
            model=str(data.get("model", "cl100k_base")),
        )
        history = data.get("history", [])
        if isinstance(history, list):
            for item in history:
                if isinstance(item, dict) and "role" in item and "content" in item:
                    role = str(item["role"])
                    if role in {"user", "assistant"}:
                        memory._history.append({"role": role, "content": str(item["content"])})
        memory._enforce_budget()
        return memory
