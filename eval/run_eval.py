from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from eval.assistants import ask_frontier, ask_oss
from eval.judge import judge_response
from eval.selfcheck import SelfCheckGPT

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_file(name: str) -> list[dict[str, Any]]:
    with (PROMPTS_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_single_turn_prompts() -> list[dict[str, Any]]:
    factual = _load_prompt_file("factual.json")
    adversarial = _load_prompt_file("adversarial.json")
    bias = _load_prompt_file("bias.json")
    all_prompts = [*factual, *adversarial, *bias]
    single_turn = [p for p in all_prompts if not p.get("multiturn", False)]
    max_prompts = int(os.getenv("EVAL_MAX_PROMPTS", "0"))
    if max_prompts > 0:
        return single_turn[:max_prompts]
    return single_turn


async def _evaluate_prompt(
    assistant_name: str,
    prompt_obj: dict[str, Any],
    semaphore: asyncio.Semaphore,
    selfchecker: SelfCheckGPT,
) -> dict[str, Any]:
    async with semaphore:
        prompt = str(prompt_obj.get("prompt", ""))
        category = str(prompt_obj.get("category", ""))
        expected = str(prompt_obj.get("expected_behavior", ""))
        start = time.perf_counter()

        assistant_error: str | None = None
        try:
            if assistant_name == "oss":
                response = await ask_oss(prompt)
            else:
                response = await ask_frontier(prompt)
        except Exception as exc:
            assistant_error = f"{type(exc).__name__}: {exc}"
            response = (
                "Assistant call failed while running this benchmark case. "
                "Response omitted due to runtime error."
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        selfcheck_result = None
        if category == "factual" and assistant_error is None:
            selfcheck_result = await selfchecker.score(
                prompt,
                response,
                ask_oss if assistant_name == "oss" else ask_frontier,
            )

        judgement = await judge_response(
            prompt=prompt,
            expected_behavior=expected,
            model_response=response,
            category=category,
        )
        return {
            "id": prompt_obj.get("id"),
            "category": category,
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "tokens": {"total_tokens": max(1, (len(prompt) + len(response)) // 4)},
            "assistant_error": assistant_error,
            "guardrail_events": [],
            "selfcheck": None
            if selfcheck_result is None
            else {
                "consistency_score": selfcheck_result.consistency_score,
                "samples": selfcheck_result.samples,
                "verdict": selfcheck_result.verdict,
            },
            "judge": judgement,
        }


async def evaluate_assistant(assistant_name: str) -> list[dict[str, Any]]:
    prompts = load_single_turn_prompts()
    sem = asyncio.Semaphore(int(os.getenv("EVAL_CONCURRENCY", "3")))
    selfchecker = SelfCheckGPT(
        n_samples=int(os.getenv("SELFCHECK_N_SAMPLES", "3")),
        method="nli",
    )
    tasks = [_evaluate_prompt(assistant_name, prompt, sem, selfchecker) for prompt in prompts]
    return await asyncio.gather(*tasks)


async def main() -> None:
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(Path(__file__).parent / "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    oss_results, frontier_results = await asyncio.gather(
        evaluate_assistant("oss"),
        evaluate_assistant("frontier"),
    )
    (output_dir / "oss_results.json").write_text(json.dumps(oss_results, indent=2), encoding="utf-8")
    (output_dir / "frontier_results.json").write_text(
        json.dumps(frontier_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
