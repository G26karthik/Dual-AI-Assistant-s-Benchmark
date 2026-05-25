from __future__ import annotations

import pytest


def test_cost_model_reports_actual_and_equivalent_costs() -> None:
    from core.observability.cost import estimate_cost

    frontier = estimate_cost(
        model_name="~openai/gpt-mini-latest",
        provider="openrouter",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    oss_free = estimate_cost(
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        provider="openrouter",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    assert frontier.actual_cost_usd > 0
    assert frontier.equivalent_cost_usd >= frontier.actual_cost_usd
    assert frontier.is_free_tier is False

    assert oss_free.actual_cost_usd == 0.0
    assert oss_free.equivalent_cost_usd > 0
    assert oss_free.is_free_tier is True


def test_aggregate_panel_judgements_uses_mean_majority_and_agreement() -> None:
    from eval.judge_panel import aggregate_panel_judgements

    aggregated = aggregate_panel_judgements(
        [
            {
                "judge_model": "judge-a",
                "scores": {"accuracy": 5, "helpfulness": 4, "safety": 5},
                "reasoning": "solid",
                "verdict": "PASS",
            },
            {
                "judge_model": "judge-b",
                "scores": {"accuracy": 3, "helpfulness": 2, "safety": 4},
                "reasoning": "mixed",
                "verdict": "FAIL",
            },
            {
                "judge_model": "judge-c",
                "scores": {"accuracy": 4, "helpfulness": 3, "safety": 5},
                "reasoning": "good enough",
                "verdict": "PASS",
            },
        ]
    )

    assert aggregated["verdict"] == "PASS"
    assert aggregated["majority_verdict"] == "PASS"
    assert aggregated["agreement_rate"] == pytest.approx(2 / 3, rel=1e-6)
    assert aggregated["scores"]["accuracy"] == pytest.approx(4.0, rel=1e-6)
    assert aggregated["scores"]["helpfulness"] == pytest.approx(3.0, rel=1e-6)
    assert aggregated["panel"][0]["judge_model"] == "judge-a"


@pytest.mark.asyncio
async def test_judge_response_panel_falls_back_per_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eval import judge_panel

    async def _fake_or_judge(*args, **kwargs):
        _ = (args, kwargs)
        return {
            "judge_model": "openrouter-a",
            "scores": {"accuracy": 4, "helpfulness": 4, "safety": 5},
            "reasoning": "clear",
            "verdict": "PASS",
        }

    async def _fake_qwen_judge(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("temporary judge failure")

    async def _fake_gemini_judge(*args, **kwargs):
        _ = (args, kwargs)
        return {
            "judge_model": "gemini-flash",
            "scores": {"accuracy": 2, "helpfulness": 3, "safety": 5},
            "reasoning": "too soft on facts",
            "verdict": "FAIL",
        }

    monkeypatch.setattr(judge_panel, "_judge_with_openrouter_llama", _fake_or_judge)
    monkeypatch.setattr(judge_panel, "_judge_with_openrouter_qwen", _fake_qwen_judge)
    monkeypatch.setattr(judge_panel, "_judge_with_gemini_flash", _fake_gemini_judge)

    result = await judge_panel.judge_response_panel(
        prompt="What causes eclipses?",
        expected_behavior="Give a correct scientific explanation.",
        model_response="The moon sometimes blocks the sun.",
        category="factual",
    )

    assert len(result["panel"]) == 3
    failed = next(entry for entry in result["panel"] if entry["judge_model"] == "openrouter-qwen3-free")
    assert failed["verdict"] == "PARTIAL"
    assert "judge failure" in failed["reasoning"].lower()
    assert "agreement_rate" in result
