from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Any

from eval.benchmarks.bold import load_bold_rows
from eval.benchmarks.jailbreak_bench import load_jailbreak_bench_rows
from eval.benchmarks.real_toxicity import load_real_toxicity_rows
from eval.benchmarks.toxigen import load_toxigen_rows
from eval.benchmarks.truthful_qa import load_truthful_qa_rows

BenchmarkLoader = Callable[[], list[dict[str, Any]]]

_DEFAULT_SAMPLE_SIZE = 100
_DEFAULT_SEEDS: dict[str, int] = {
    "truthful_qa": 101,
    "toxigen": 202,
    "bold": 303,
    "real_toxicity": 404,
    "jailbreak_bench": 505,
}


def _sample_size() -> int:
    try:
        value = int(os.getenv("EVAL_BENCHMARK_SAMPLE_SIZE", str(_DEFAULT_SAMPLE_SIZE)))
    except ValueError:
        return _DEFAULT_SAMPLE_SIZE
    return max(1, value)


def _benchmark_seed(benchmark_name: str) -> int:
    env_name = f"EVAL_BENCHMARK_SEED_{benchmark_name.upper()}"
    try:
        return int(os.getenv(env_name, str(_DEFAULT_SEEDS[benchmark_name])))
    except ValueError:
        return _DEFAULT_SEEDS[benchmark_name]


def _select_rows(
    benchmark_name: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sample_size = _sample_size()
    if len(rows) <= sample_size:
        return sorted(rows, key=lambda item: str(item.get("id", "")))
    rng = random.Random(_benchmark_seed(benchmark_name))
    selected = rng.sample(rows, sample_size)
    return sorted(selected, key=lambda item: str(item.get("id", "")))


def load_public_benchmarks() -> list[dict[str, Any]]:
    benchmark_loaders: tuple[tuple[str, BenchmarkLoader], ...] = (
        ("truthful_qa", load_truthful_qa_rows),
        ("toxigen", load_toxigen_rows),
        ("bold", load_bold_rows),
        ("real_toxicity", load_real_toxicity_rows),
        ("jailbreak_bench", load_jailbreak_bench_rows),
    )
    combined: list[dict[str, Any]] = []
    for benchmark_name, loader in benchmark_loaders:
        rows = loader()
        combined.extend(_select_rows(benchmark_name, rows))
    return combined
