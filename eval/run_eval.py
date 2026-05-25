from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.assistants import (
    ask_frontier,
    ask_frontier_with_meta,
    ask_oss,
    ask_oss_with_meta,
)
from eval.benchmarks import load_public_benchmarks
from eval.judge_panel import judge_response_panel
from eval.selfcheck import SelfCheckGPT

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_file(name: str) -> list[dict[str, Any]]:
    with (PROMPTS_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_legacy_single_turn_prompts() -> list[dict[str, Any]]:
    factual = _load_prompt_file("factual.json")
    adversarial = _load_prompt_file("adversarial.json")
    bias = _load_prompt_file("bias.json")
    all_prompts = [*factual, *adversarial, *bias]
    single_turn = [p for p in all_prompts if not p.get("multiturn", False)]
    max_prompts = int(os.getenv("EVAL_MAX_PROMPTS", "0"))
    if max_prompts > 0:
        return single_turn[:max_prompts]
    return single_turn


def load_single_turn_prompts() -> list[dict[str, Any]]:
    prompts = load_public_benchmarks()
    if os.getenv("EVAL_INCLUDE_LEGACY_PROMPTS", "false").strip().lower() in {"1", "true", "yes"}:
        prompts.extend(_load_legacy_single_turn_prompts())
    return prompts


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
                assistant_result = await ask_oss_with_meta(prompt)
            else:
                assistant_result = await ask_frontier_with_meta(prompt)
        except Exception as exc:
            assistant_error = f"{type(exc).__name__}: {exc}"
            response = (
                "Assistant call failed while running this benchmark case. "
                "Response omitted due to runtime error."
            )
            assistant_result = None

        latency_ms = int((time.perf_counter() - start) * 1000)
        response = response if assistant_result is None else assistant_result.text
        selfcheck_result = None
        judge_task = judge_response_panel(
            prompt=prompt,
            expected_behavior=expected,
            model_response=response,
            category=category,
        )
        if category == "factual" and assistant_error is None:
            selfcheck_task = selfchecker.score(
                prompt,
                response,
                ask_oss if assistant_name == "oss" else ask_frontier,
            )
            selfcheck_result, judgement = await asyncio.gather(selfcheck_task, judge_task)
        else:
            judgement = await judge_task
        assistant_cost = (
            {"actual_cost_usd": 0.0, "equivalent_cost_usd": 0.0}
            if assistant_result is None
            else assistant_result.cost
        )
        return {
            "id": prompt_obj.get("id"),
            "benchmark_name": prompt_obj.get("benchmark_name"),
            "source_dataset": prompt_obj.get("source_dataset"),
            "category": category,
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "tokens": (
                {"total_tokens": max(1, (len(prompt) + len(response)) // 4)}
                if assistant_result is None
                else assistant_result.tokens
            ),
            "assistant_model": None if assistant_result is None else assistant_result.model_name,
            "assistant_provider": None if assistant_result is None else assistant_result.provider,
            "assistant_cost": assistant_cost,
            "assistant_error": assistant_error,
            "guardrail_events": [] if assistant_result is None else assistant_result.guardrail_events,
            "selfcheck": None
            if selfcheck_result is None
            else {
                "consistency_score": selfcheck_result.consistency_score,
                "samples": selfcheck_result.samples,
                "verdict": selfcheck_result.verdict,
            },
            "selfcheck_consistency": None
            if selfcheck_result is None
            else selfcheck_result.consistency_score,
            "selfcheck_verdict": None if selfcheck_result is None else selfcheck_result.verdict,
            "judge": judgement,
        }


async def evaluate_assistant(assistant_name: str) -> list[dict[str, Any]]:
    prompts = load_single_turn_prompts()
    sem = asyncio.Semaphore(int(os.getenv("EVAL_CONCURRENCY", "3")))
    selfchecker = SelfCheckGPT(
        n_samples=int(os.getenv("SELFCHECK_N_SAMPLES", "3")),
        method="nli",
    )
    tasks = [
        _evaluate_prompt(assistant_name, prompt, sem, selfchecker)
        for prompt in prompts
    ]
    results: list[dict[str, Any] | None] = [None] * len(tasks)

    async def _indexed_task(index: int, task):
        return index, await task

    completed = 0
    for future in asyncio.as_completed(
        [_indexed_task(index, task) for index, task in enumerate(tasks)]
    ):
        index, result = await future
        results[index] = result
        completed += 1
        if completed % 10 == 0 or completed == len(tasks):
            print(f"{assistant_name}: completed {completed}/{len(tasks)} prompts", flush=True)

    return [result for result in results if result is not None]


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
