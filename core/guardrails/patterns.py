from __future__ import annotations

PROMPT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+your\s+system\s+prompt",
    r"you\s+are\s+now",
    r"\bdan\b",
    r"\bjailbreak\b",
    r"act\s+as\s+if",
    r"pretend\s+you\s+have\s+no",
    r"\bsystem\s*:",
    r"1gn0r3\s+4ll\s+pr3v10us",
    r"SWdub3Jl",
]

PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(?:\+?\d{1,3})?[-.\s(]?\d{3}[-.\s)]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}
