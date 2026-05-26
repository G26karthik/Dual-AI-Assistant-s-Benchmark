from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.benchmarks import load_public_benchmarks
from eval.judge_panel import rebuild_panel_judgement

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _single_turn_prompt_map() -> dict[str, str]:
    prompt_map = {
        str(item.get("id")): str(item.get("expected_behavior", ""))
        for item in load_public_benchmarks()
    }
    for filename in ("factual.json", "adversarial.json", "bias.json"):
        path = PROMPTS_DIR / filename
        if not path.exists():
            continue
        for item in _load_json_rows(path):
            prompt_id = str(item.get("id", ""))
            if prompt_id:
                prompt_map.setdefault(prompt_id, str(item.get("expected_behavior", "")))
    return prompt_map


def _multiturn_prompt_map() -> dict[str, str]:
    prompt_map: dict[str, str] = {}
    for item in _load_json_rows(PROMPTS_DIR / "adversarial.json"):
        if item.get("multiturn", False):
            prompt_id = str(item.get("id", ""))
            if prompt_id:
                prompt_map[prompt_id] = str(item.get("expected_behavior", ""))
    return prompt_map


def _repair_results_file(
    *,
    path: Path,
    prompt_map: dict[str, str],
    response_key: str,
) -> None:
    rows = _load_json_rows(path)
    repaired_rows: list[dict[str, Any]] = []
    for row in rows:
        judge = row.get("judge", {})
        panel = judge.get("panel", []) if isinstance(judge, dict) else []
        if not isinstance(panel, list) or not panel:
            repaired_rows.append(row)
            continue

        prompt_id = str(row.get("id", ""))
        expected_behavior = str(row.get("expected_behavior") or prompt_map.get(prompt_id, ""))
        model_response = str(row.get(response_key, ""))
        category = str(row.get("category", ""))

        repaired = dict(row)
        repaired["judge"] = rebuild_panel_judgement(
            panel=panel,
            category=category,
            expected_behavior=expected_behavior,
            model_response=model_response,
        )
        repaired_rows.append(repaired)

    path.write_text(json.dumps(repaired_rows, indent=2), encoding="utf-8")


def main() -> None:
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(Path(__file__).parent / "results")))
    single_turn_prompts = _single_turn_prompt_map()
    multiturn_prompts = _multiturn_prompt_map()

    _repair_results_file(
        path=output_dir / "oss_results.json",
        prompt_map=single_turn_prompts,
        response_key="response",
    )
    _repair_results_file(
        path=output_dir / "frontier_results.json",
        prompt_map=single_turn_prompts,
        response_key="response",
    )
    _repair_results_file(
        path=output_dir / "oss_multiturn_results.json",
        prompt_map=multiturn_prompts,
        response_key="final_response",
    )
    _repair_results_file(
        path=output_dir / "frontier_multiturn_results.json",
        prompt_map=multiturn_prompts,
        response_key="final_response",
    )


if __name__ == "__main__":
    main()
