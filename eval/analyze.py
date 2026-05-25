from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_scores(results: list[dict[str, Any]], model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        judge = item.get("judge", {})
        scores = judge.get("scores", {}) if isinstance(judge, dict) else {}
        raw_verdict = judge.get("verdict", "UNKNOWN") if isinstance(judge, dict) else "UNKNOWN"
        verdict = str(raw_verdict).strip().upper()
        if verdict not in {"PASS", "PARTIAL", "FAIL"}:
            verdict = "PARTIAL"
        tokens = (
            item.get("tokens", {}).get("total_tokens", 0)
            if isinstance(item.get("tokens"), dict)
            else 0
        )
        token_count = int(tokens)
        assistant_cost = item.get("assistant_cost", {}) if isinstance(item.get("assistant_cost"), dict) else {}
        judge_cost = judge.get("cost", {}) if isinstance(judge, dict) else {}
        assistant_actual_cost = float(assistant_cost.get("actual_cost_usd", 0.0))
        assistant_equivalent_cost = float(assistant_cost.get("equivalent_cost_usd", 0.0))
        judge_actual_cost = float(judge_cost.get("actual_cost_usd", 0.0))
        judge_equivalent_cost = float(judge_cost.get("equivalent_cost_usd", 0.0))
        rows.append(
            {
                "model": model_name,
                "id": item.get("id"),
                "benchmark_name": item.get("benchmark_name"),
                "source_dataset": item.get("source_dataset"),
                "category": item.get("category"),
                "verdict": verdict,
                "majority_verdict": judge.get("majority_verdict", verdict) if isinstance(judge, dict) else verdict,
                "agreement_rate": judge.get("agreement_rate") if isinstance(judge, dict) else None,
                "latency_ms": item.get("latency_ms", 0),
                "tokens": token_count,
                "assistant_actual_cost_usd": assistant_actual_cost,
                "assistant_equivalent_cost_usd": assistant_equivalent_cost,
                "judge_actual_cost_usd": judge_actual_cost,
                "judge_equivalent_cost_usd": judge_equivalent_cost,
                "actual_cost_usd": assistant_actual_cost + judge_actual_cost,
                "equivalent_cost_usd": assistant_equivalent_cost + judge_equivalent_cost,
                "accuracy": scores.get("accuracy") if isinstance(scores, dict) else None,
                "hallucination_resistance": scores.get("hallucination_resistance")
                if isinstance(scores, dict)
                else None,
                "safety": scores.get("safety") if isinstance(scores, dict) else None,
                "bias_score": scores.get("bias_score") if isinstance(scores, dict) else None,
                "refusal_quality": scores.get("refusal_quality") if isinstance(scores, dict) else None,
                "helpfulness": scores.get("helpfulness") if isinstance(scores, dict) else None,
                "selfcheck_consistency": item.get("selfcheck", {}).get("consistency_score")
                if isinstance(item.get("selfcheck"), dict)
                else None,
                "selfcheck_verdict": item.get("selfcheck", {}).get("verdict")
                if isinstance(item.get("selfcheck"), dict)
                else item.get("selfcheck_verdict"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(Path(__file__).parent / "results")))
    oss = _load_results(output_dir / "oss_results.json")
    frontier = _load_results(output_dir / "frontier_results.json")
    df = pd.concat(
        [
            _flatten_scores(oss, "oss"),
            _flatten_scores(frontier, "frontier"),
        ],
        ignore_index=True,
    )
    verdict_counts = {
        f"{model}:{verdict}": int(count)
        for (model, verdict), count in df.groupby(["model", "verdict"]).size().to_dict().items()
    }
    summary = {
        "rows": int(len(df)),
        "by_model": df.groupby("model").size().to_dict(),
        "by_benchmark": {
            f"{model}:{benchmark}": int(count)
            for (model, benchmark), count in df.groupby(["model", "benchmark_name"]).size().to_dict().items()
        },
        "avg_latency_ms": df.groupby("model")["latency_ms"].mean().fillna(0).to_dict(),
        "p50_latency_ms": df.groupby("model")["latency_ms"].median().fillna(0).to_dict(),
        "p95_latency_ms": df.groupby("model")["latency_ms"].quantile(0.95).fillna(0).to_dict(),
        "avg_tokens": df.groupby("model")["tokens"].mean().fillna(0).to_dict(),
        "actual_total_cost_usd": df.groupby("model")["actual_cost_usd"].sum().fillna(0).to_dict(),
        "equivalent_total_cost_usd": df.groupby("model")["equivalent_cost_usd"].sum().fillna(0).to_dict(),
        "avg_agreement_rate": df.groupby("model")["agreement_rate"].mean().fillna(0).to_dict(),
        "verdict_counts": verdict_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(output_dir / "flattened_scores.csv", index=False)


if __name__ == "__main__":
    main()
