from __future__ import annotations

import os
import re
from typing import Any

from huggingface_hub import AsyncInferenceClient

_HEURISTIC_PATTERNS: tuple[str, ...] = (
    r"\bkill\s+(?:all|those|them)\b",
    r"\bhate(?:ful)?\b",
    r"\bslur\b",
    r"\bgenocide\b",
    r"\blynch\b",
    r"\bvermin\b",
    r"\bsubhuman\b",
)


def _heuristic_toxicity(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for pattern in _HEURISTIC_PATTERNS if re.search(pattern, lowered))
    if hits == 0:
        return 0.0
    return min(1.0, 0.35 + (hits * 0.25))


def _extract_toxic_score(result: Any) -> float:
    if isinstance(result, list):
        best = 0.0
        for item in result:
            best = max(best, _extract_toxic_score(item))
        return best
    if isinstance(result, dict):
        label = str(result.get("label", "")).lower()
        score = float(result.get("score", 0.0))
        if any(token in label for token in ("toxic", "hate", "insult", "abuse", "threat")):
            return score
        return 0.0
    label = str(getattr(result, "label", "")).lower()
    score = float(getattr(result, "score", 0.0))
    if any(token in label for token in ("toxic", "hate", "insult", "abuse", "threat")):
        return score
    return 0.0


async def classify_toxicity(text: str) -> float:
    if not text.strip():
        return 0.0
    model_id = os.getenv("TOXICITY_MODEL_ID", "unitary/unbiased-toxic-roberta").strip()
    token = (os.getenv("HF_INFERENCE_TOKEN") or os.getenv("HF_TOKEN") or "").strip() or None
    try:
        client = AsyncInferenceClient(model=model_id, token=token)
        classify = getattr(client, "text_classification", None)
        if classify is None:
            return _heuristic_toxicity(text)
        result = await classify(text, top_k=None)
        return max(_heuristic_toxicity(text), _extract_toxic_score(result))
    except Exception:
        return _heuristic_toxicity(text)
