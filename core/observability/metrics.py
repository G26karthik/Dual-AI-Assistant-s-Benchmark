from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record_turn(self, log_entry: dict[str, Any]) -> None:
        self.entries.append(log_entry)

    def summary(self) -> dict[str, float | int]:
        if not self.entries:
            return {
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "guardrail_blocks": 0,
                "tool_call_count": 0,
                "sessions_count": 0,
            }

        latencies = [
            float(entry.get("latency_ms", {}).get("total", 0.0))
            for entry in self.entries
            if isinstance(entry.get("latency_ms"), dict)
        ]
        tokens = [
            int(entry.get("tokens", {}).get("total_tokens", 0))
            for entry in self.entries
            if isinstance(entry.get("tokens"), dict)
        ]
        cost = float(sum(float(entry.get("estimated_cost_usd", 0.0)) for entry in self.entries))
        blocks = 0
        tool_calls = 0
        sessions = set()
        for entry in self.entries:
            input_guard = entry.get("input_guard", {})
            output_guard = entry.get("output_guard", {})
            if isinstance(input_guard, dict) and not input_guard.get("allowed", True):
                blocks += 1
            if isinstance(output_guard, dict) and not output_guard.get("allowed", True):
                blocks += 1
            tool_calls += len(entry.get("tool_calls", [])) if isinstance(entry.get("tool_calls"), list) else 0
            if "session_id" in entry:
                sessions.add(entry["session_id"])

        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
        return {
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "total_tokens": sum(tokens),
            "estimated_cost_usd": cost,
            "guardrail_blocks": blocks,
            "tool_call_count": tool_calls,
            "sessions_count": len(sessions),
        }
