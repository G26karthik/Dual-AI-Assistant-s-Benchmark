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
        estimated_cost = 0.0 if model_name == "oss" else (token_count / 1000.0) * 0.01
        rows.append(
            {
                "model": model_name,
                "id": item.get("id"),
                "category": item.get("category"),
                "verdict": verdict,
                "latency_ms": item.get("latency_ms", 0),
                "tokens": token_count,
                "estimated_cost_usd": estimated_cost,
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
        "avg_latency_ms": df.groupby("model")["latency_ms"].mean().fillna(0).to_dict(),
        "p50_latency_ms": df.groupby("model")["latency_ms"].median().fillna(0).to_dict(),
        "p95_latency_ms": df.groupby("model")["latency_ms"].quantile(0.95).fillna(0).to_dict(),
        "avg_tokens": df.groupby("model")["tokens"].mean().fillna(0).to_dict(),
        "estimated_total_cost_usd": df.groupby("model")["estimated_cost_usd"].sum().fillna(0).to_dict(),
        "verdict_counts": verdict_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(output_dir / "flattened_scores.csv", index=False)


if __name__ == "__main__":
    main()
