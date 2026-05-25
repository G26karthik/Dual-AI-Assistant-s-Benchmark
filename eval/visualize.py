from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _to_frame(results: list[dict[str, Any]], model: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in results:
        scores = r.get("judge", {}).get("scores", {})
        rows.append(
            {
                "model": model,
                "id": r.get("id"),
                "category": r.get("category"),
                "verdict": r.get("judge", {}).get("verdict"),
                "latency_ms": r.get("latency_ms", 0),
                "accuracy": scores.get("accuracy"),
                "hallucination_resistance": scores.get("hallucination_resistance"),
                "safety": scores.get("safety"),
                "bias_score": scores.get("bias_score"),
                "refusal_quality": scores.get("refusal_quality"),
                "helpfulness": scores.get("helpfulness"),
                "selfcheck_consistency": r.get("selfcheck", {}).get("consistency_score")
                if isinstance(r.get("selfcheck"), dict)
                else None,
            }
        )
    return pd.DataFrame(rows)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    results_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(Path(__file__).parent / "results")))
    assets_dir = Path(__file__).resolve().parents[1] / "report" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    oss = _to_frame(_load(results_dir / "oss_results.json"), "oss")
    frontier = _to_frame(_load(results_dir / "frontier_results.json"), "frontier")
    df = pd.concat([oss, frontier], ignore_index=True)
    if df.empty:
        return

    dims = ["accuracy", "hallucination_resistance", "safety", "bias_score", "refusal_quality", "helpfulness"]
    radar_df = df.groupby("model")[dims].mean(numeric_only=True).fillna(0)
    angles = [n / float(len(dims)) * 2 * 3.14159 for n in range(len(dims))]
    angles += angles[:1]
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for model, row in radar_df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_title("Model Radar Scores")
    ax.legend()
    _savefig(assets_dir / "radar_chart.png")

    cat_df = df.melt(id_vars=["model", "category"], value_vars=dims, var_name="dimension", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=cat_df, x="category", y="score", hue="model")
    plt.title("Grouped Category Scores")
    _savefig(assets_dir / "grouped_bar.png")

    pivot = df.pivot_table(index="id", columns="model", values="safety", aggfunc="mean").fillna(0)
    plt.figure(figsize=(8, 10))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False)
    plt.title("Safety Heatmap")
    _savefig(assets_dir / "safety_heatmap.png")

    latency = df.groupby("model")["latency_ms"].agg(["mean", "median", "max"]).reset_index()
    plt.figure(figsize=(8, 5))
    latency.set_index("model")[["mean", "median", "max"]].plot(kind="bar")
    plt.ylabel("Latency ms")
    plt.title("Latency Metrics")
    _savefig(assets_dir / "latency_cost_bar.png")

    verdicts = df.groupby(["model", "category", "verdict"]).size().reset_index(name="count")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=verdicts, x="category", y="count", hue="verdict")
    plt.title("Pass/Partial/Fail Summary")
    _savefig(assets_dir / "pass_fail_summary.png")

    factual = df[df["category"] == "factual"].dropna(subset=["selfcheck_consistency", "hallucination_resistance"])
    if not factual.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=factual,
            x="selfcheck_consistency",
            y="hallucination_resistance",
            hue="model",
        )
        plt.title("SelfCheck Consistency vs Hallucination Resistance")
        _savefig(assets_dir / "selfcheck_scatter.png")


if __name__ == "__main__":
    main()
