from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


def _clean_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class _SpanState:
    name: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str = "DEFAULT"


class BaseSpan:
    is_noop = False

    def __init__(self, state: _SpanState) -> None:
        self._state = state

    def update(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        if input is not None:
            self._state.input = _clean_payload(input)
        if output is not None:
            self._state.output = _clean_payload(output)
        if metadata:
            self._state.metadata.update(_clean_payload(metadata))
        if level is not None:
            self._state.level = level

    def record_error(self, error: BaseException | str) -> None:
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self.update(metadata={"error": message}, level="ERROR")

    def __enter__(self) -> BaseSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.record_error(exc)
        _ = (exc_type, tb)


class NoopSpan(BaseSpan):
    is_noop = True


class LangfuseSpan(BaseSpan):
    def __init__(
        self,
        *,
        client: Any,
        kind: str,
        name: str,
        model: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            _SpanState(
                name=name,
                input=_clean_payload(input),
                metadata=_clean_payload(metadata) if metadata else {},
            )
        )
        self._client = client
        self._kind = kind
        self._model = model
        self._handle: Any | None = None

    def __enter__(self) -> LangfuseSpan:
        try:
            if self._kind == "generation" and hasattr(self._client, "generation"):
                self._handle = self._client.generation(
                    name=self._state.name,
                    model=self._model,
                    input=self._state.input,
                    metadata=self._state.metadata or None,
                )
            elif hasattr(self._client, "trace"):
                self._handle = self._client.trace(
                    name=self._state.name,
                    input=self._state.input,
                    metadata=self._state.metadata or None,
                )
            elif hasattr(self._client, "span"):
                self._handle = self._client.span(
                    name=self._state.name,
                    input=self._state.input,
                    metadata=self._state.metadata or None,
                )
        except Exception:
            self._handle = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc, tb)
        if self._handle is None:
            return
        try:
            if hasattr(self._handle, "update"):
                self._handle.update(
                    input=self._state.input,
                    output=self._state.output,
                    metadata=self._state.metadata or None,
                    level=self._state.level,
                )
            if hasattr(self._handle, "end"):
                self._handle.end(output=self._state.output)
        except Exception:
            return


class BaseTracer:
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSpan:
        raise NotImplementedError

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSpan:
        raise NotImplementedError

    def flush(self) -> None:
        return None


class NoopTracer(BaseTracer):
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSpan:
        return NoopSpan(_SpanState(name=name, input=input, metadata=metadata or {}))

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSpan:
        _ = model
        return NoopSpan(_SpanState(name=name, input=input, metadata=metadata or {}))


class LangfuseTracer(BaseTracer):
    def __init__(self, client: Any) -> None:
        self._client = client

    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSpan:
        return LangfuseSpan(
            client=self._client,
            kind="span",
            name=name,
            input=input,
            metadata=metadata,
        )

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSpan:
        return LangfuseSpan(
            client=self._client,
            kind="generation",
            name=name,
            model=model,
            input=input,
            metadata=metadata,
        )

    def flush(self) -> None:
        if hasattr(self._client, "flush"):
            try:
                self._client.flush()
            except Exception:
                return


_TRACER_LOCK = threading.Lock()
_TRACER: BaseTracer | None = None


def _build_tracer() -> BaseTracer:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = (
        os.getenv("LANGFUSE_HOST", "").strip()
        or os.getenv("LANGFUSE_BASE_URL", "").strip()
    )
    if not public_key or not secret_key:
        return NoopTracer()
    try:
        langfuse_module = importlib.import_module("langfuse")
        langfuse_ctor = langfuse_module.Langfuse
        client = langfuse_ctor(
            public_key=public_key,
            secret_key=secret_key,
            host=host or None,
        )
    except Exception:
        return NoopTracer()
    return LangfuseTracer(client)


def get_tracer(*, force_rebuild: bool = False) -> BaseTracer:
    global _TRACER
    if force_rebuild:
        with _TRACER_LOCK:
            _TRACER = _build_tracer()
            return _TRACER
    if _TRACER is None:
        with _TRACER_LOCK:
            if _TRACER is None:
                _TRACER = _build_tracer()
    return _TRACER
