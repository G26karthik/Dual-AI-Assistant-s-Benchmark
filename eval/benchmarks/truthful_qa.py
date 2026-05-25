from __future__ import annotations

import os
from typing import Any

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional until deps are installed
    load_dataset = None  # type: ignore[assignment]


def _require_datasets() -> Any:
    if load_dataset is None:
        raise RuntimeError("The datasets package is required for public benchmark loading.")
    return load_dataset


def _pick_truthful_answer(row: dict[str, Any]) -> str:
    mc1_targets = row.get("mc1_targets")
    if isinstance(mc1_targets, dict):
        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if isinstance(choices, list) and isinstance(labels, list):
            for choice, label in zip(choices, labels, strict=False):
                if int(label) == 1:
                    return str(choice)
    choices = row.get("choices", [])
    label = row.get("label")
    if isinstance(choices, list) and label is not None:
        try:
            return str(choices[int(label)])
        except (ValueError, TypeError, IndexError):
            return ""
    return str(row.get("best_answer", ""))


def load_truthful_qa_rows() -> list[dict[str, Any]]:
    dataset_loader = _require_datasets()
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    dataset = None
    for repo_id, config in (
        ("truthfulqa/truthful_qa", "multiple_choice"),
        ("EleutherAI/truthful_qa_mc", "multiple_choice"),
    ):
        try:
            dataset = dataset_loader(
                repo_id,
                config,
                split="validation",
                token=token,
            )
            break
        except Exception:
            dataset = None
    if dataset is None:
        raise RuntimeError("Could not load a public TruthfulQA dataset or mirror.")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        truthful_answer = _pick_truthful_answer(row)
        rows.append(
            {
                "id": f"truthful_qa-{index}",
                "benchmark_name": "truthful_qa",
                "source_dataset": "truthfulqa/truthful_qa",
                "category": "factual",
                "prompt": question,
                "expected_behavior": (
                    "Answer the question truthfully, correct false premises, and prefer the "
                    f"known truthful answer when one is provided: {truthful_answer}"
                ).strip(),
                "reference_answer": truthful_answer,
                "metadata": {"question_category": row.get("category")},
            }
        )
    return rows
