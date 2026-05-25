from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]


@dataclass
class SelfCheckResult:
    consistency_score: float
    samples: list[str]
    verdict: Literal["consistent", "inconsistent", "uncertain"]


class SelfCheckGPT:
    def __init__(self, n_samples: int = 3, method: Literal["bertscore", "nli"] = "nli") -> None:
        self.n_samples = n_samples
        self.method = method
        self._nli_model = None
        if method == "nli" and CrossEncoder is not None:
            self._nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    @staticmethod
    def _lexical_overlap(a: str, b: str) -> float:
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def _pair_score(self, a: str, b: str) -> float:
        if self._nli_model is None:
            return self._lexical_overlap(a, b)
        logits = self._nli_model.predict([(a, b)])
        # Cross-encoder nli returns [contradiction, entailment, neutral] in common configs.
        entailment = float(logits[0][1]) if len(logits[0]) > 1 else 0.0
        contradiction = float(logits[0][0]) if len(logits[0]) > 0 else 0.0
        score = (entailment - contradiction + 1.0) / 2.0
        return max(0.0, min(1.0, score))

    async def score(
        self,
        prompt: str,
        primary_response: str,
        assistant_fn,
    ) -> SelfCheckResult:
        samples: list[str] = []
        try:
            for _ in range(self.n_samples):
                sampled = await assistant_fn(prompt)
                samples.append(sampled)
        except Exception:
            return SelfCheckResult(consistency_score=0.5, samples=samples, verdict="uncertain")

        all_responses = [primary_response, *samples]
        pairs = list(combinations(all_responses, 2))
        if not pairs:
            return SelfCheckResult(consistency_score=1.0, samples=samples, verdict="consistent")

        scores = [await asyncio.to_thread(self._pair_score, a, b) for a, b in pairs]
        consistency = sum(scores) / len(scores)
        if consistency > 0.8:
            verdict = "consistent"
        elif consistency < 0.5:
            verdict = "inconsistent"
        else:
            verdict = "uncertain"
        return SelfCheckResult(consistency_score=consistency, samples=samples, verdict=verdict)
