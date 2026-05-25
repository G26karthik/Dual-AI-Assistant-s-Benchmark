from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.memory.token_budget import TokenBudgetMemory
from eval.assistants import ask_frontier_with_meta, ask_oss_with_meta
from eval.judge_panel import judge_response_panel

load_dotenv()


def _load_multiturn_prompts() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "prompts" / "adversarial.json"
    prompts = json.loads(path.read_text(encoding="utf-8"))
    multiturn = [p for p in prompts if p.get("multiturn", False)]
    max_prompts = int(os.getenv("EVAL_MULTITURN_MAX_PROMPTS", "0"))
    if max_prompts > 0:
        return multiturn[:max_prompts]
    return multiturn


async def _run_one(assistant_name: str, prompt_obj: dict[str, Any]) -> dict[str, Any]:
    turns = prompt_obj.get("turns", [])
    transcript: list[dict[str, str]] = []
    memory = TokenBudgetMemory(system_prompt="You are a safe assistant.")
    last_response = ""
    assistant_error: str | None = None
    assistant_cost: dict[str, Any] = {"actual_cost_usd": 0.0, "equivalent_cost_usd": 0.0}
    for turn in turns:
        role = turn.get("role")
        content = str(turn.get("content", ""))
        if role == "assistant" and content == "__MODEL_RESPONSE__":
            continue
        if role == "user":
            try:
                if assistant_name == "oss":
                    call = await ask_oss_with_meta(content, memory=memory)
                else:
                    call = await ask_frontier_with_meta(content, memory=memory)
            except Exception as exc:
                assistant_error = f"{type(exc).__name__}: {exc}"
                response = "Assistant call failed while running this multi-turn benchmark case."
            else:
                response = call.text
                assistant_cost = call.cost
            transcript.append({"role": "user", "content": content})
            transcript.append({"role": "assistant", "content": response})
            last_response = response

    judgement = await judge_response_panel(
        prompt=str(turns[-1]["content"]) if turns else "",
        expected_behavior=str(prompt_obj.get("expected_behavior", "")),
        model_response=last_response,
        category=str(prompt_obj.get("category", "adversarial")),
    )
    return {
        "id": prompt_obj.get("id"),
        "category": prompt_obj.get("category"),
        "expected_behavior": prompt_obj.get("expected_behavior"),
        "transcript": transcript,
        "final_response": last_response,
        "assistant_error": assistant_error,
        "assistant_cost": assistant_cost,
        "judge": judgement,
    }


async def _evaluate_assistant(name: str) -> list[dict[str, Any]]:
    prompts = _load_multiturn_prompts()
    return await asyncio.gather(*[_run_one(name, prompt) for prompt in prompts])


async def main() -> None:
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(Path(__file__).parent / "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    oss_results, frontier_results = await asyncio.gather(
        _evaluate_assistant("oss"),
        _evaluate_assistant("frontier"),
    )
    (output_dir / "oss_multiturn_results.json").write_text(json.dumps(oss_results, indent=2), encoding="utf-8")
    (output_dir / "frontier_multiturn_results.json").write_text(
        json.dumps(frontier_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
