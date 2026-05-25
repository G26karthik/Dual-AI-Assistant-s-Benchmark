"""Generate the one-page evaluation PDF.

Reads live numbers from ``eval/results/summary.json`` and
``eval/results/flattened_scores.csv`` (produced by ``eval/analyze.py``) and
renders a single-page letter-sized PDF using ReportLab Platypus.

Hard requirements:
- exactly one page (verified at render time and by the test suite)
- URLs are plain text only, never ``<a>``-tagged
- no timestamps, no dates, no ``Generated on`` lines, no contact metadata
- two infographics from ``report/assets/`` rendered side-by-side
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepInFrame,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

GITHUB_URL = "https://github.com/G26karthik/Dual-AI-Assistant-s-Benchmark"
HF_SPACE_URL = "https://huggingface.co/spaces/LuciferMrng/dual-ai-assistant-benchmark-oss"


# --- Data loading ----------------------------------------------------------


def _load_summary(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_scores(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "flattened_scores.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _fmt_float(value: Any, places: int = 2) -> str:
    if value is None:
        return "not measured"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "not measured"


def _safe_get_metric(summary: dict[str, Any], key: str, model: str) -> Any:
    bucket = summary.get(key, {})
    if not isinstance(bucket, dict) or model not in bucket:
        return None
    return bucket[model]


def _verdict_count(summary: dict[str, Any], model: str, verdict: str) -> int:
    counts = summary.get("verdict_counts", {})
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(f"{model}:{verdict}", 0))


def _dim_avg(df: pd.DataFrame, model: str, dim: str) -> Any:
    if df.empty or dim not in df.columns:
        return None
    subset = df[df["model"] == model][dim].dropna()
    if subset.empty:
        return None
    return float(subset.mean())


def _selfcheck_avg(df: pd.DataFrame, model: str) -> Any:
    if df.empty or "selfcheck_consistency" not in df.columns:
        return None
    subset = df[(df["model"] == model) & (df["category"] == "factual")][
        "selfcheck_consistency"
    ].dropna()
    if subset.empty:
        return None
    return float(subset.mean())


def _agreement_avg(summary: dict[str, Any], model: str) -> Any:
    return _safe_get_metric(summary, "avg_agreement_rate", model)


# --- Styles ----------------------------------------------------------------


def _make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=13,
            spaceAfter=2,
            leading=15,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#475569"),
            spaceAfter=3,
            leading=10,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9,
            spaceBefore=4,
            spaceAfter=1,
            textColor=colors.HexColor("#111827"),
            leading=11,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            spaceAfter=1,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            leftIndent=10,
            bulletIndent=0,
            spaceAfter=0,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#475569"),
            leading=8.5,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#475569"),
            leading=9,
        ),
    }


def _bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(f"&bull;&nbsp; {item}", styles["bullet"]) for item in items]


# --- KPI table -------------------------------------------------------------


def _kpi_table(
    summary: dict[str, Any], df: pd.DataFrame, styles: dict[str, ParagraphStyle]
) -> Table:
    rows: list[list[str]] = [
        ["Metric", "OSS (Qwen2.5-0.5B / LLaMA-3.2-3B free)", "Frontier (~gpt-mini-latest)"]
    ]

    rows.append(
        [
            "Latency P50 / P95 (ms)",
            f"{_fmt_float(_safe_get_metric(summary, 'p50_latency_ms', 'oss'), 0)} / "
            f"{_fmt_float(_safe_get_metric(summary, 'p95_latency_ms', 'oss'), 0)}",
            f"{_fmt_float(_safe_get_metric(summary, 'p50_latency_ms', 'frontier'), 0)} / "
            f"{_fmt_float(_safe_get_metric(summary, 'p95_latency_ms', 'frontier'), 0)}",
        ]
    )
    rows.append(
        [
            "Actual / equivalent eval cost (USD)",
            f"${_fmt_float(_safe_get_metric(summary, 'actual_total_cost_usd', 'oss'), 5)} / "
            f"${_fmt_float(_safe_get_metric(summary, 'equivalent_total_cost_usd', 'oss'), 5)}",
            f"${_fmt_float(_safe_get_metric(summary, 'actual_total_cost_usd', 'frontier'), 5)} / "
            f"${_fmt_float(_safe_get_metric(summary, 'equivalent_total_cost_usd', 'frontier'), 5)}",
        ]
    )
    rows.append(
        [
            "Verdicts (PASS / PARTIAL / FAIL)",
            f"{_verdict_count(summary, 'oss', 'PASS')} / "
            f"{_verdict_count(summary, 'oss', 'PARTIAL')} / "
            f"{_verdict_count(summary, 'oss', 'FAIL')}",
            f"{_verdict_count(summary, 'frontier', 'PASS')} / "
            f"{_verdict_count(summary, 'frontier', 'PARTIAL')} / "
            f"{_verdict_count(summary, 'frontier', 'FAIL')}",
        ]
    )
    rows.append(
        [
            "Hallucination resistance / Safety (1-5)",
            f"{_fmt_float(_dim_avg(df, 'oss', 'hallucination_resistance'), 2)} / "
            f"{_fmt_float(_dim_avg(df, 'oss', 'safety'), 2)}",
            f"{_fmt_float(_dim_avg(df, 'frontier', 'hallucination_resistance'), 2)} / "
            f"{_fmt_float(_dim_avg(df, 'frontier', 'safety'), 2)}",
        ]
    )
    rows.append(
        [
            "Panel agreement / SelfCheck",
            f"{_fmt_float(_agreement_avg(summary, 'oss'), 2)} / {_fmt_float(_selfcheck_avg(df, 'oss'), 3)}",
            f"{_fmt_float(_agreement_avg(summary, 'frontier'), 2)} / {_fmt_float(_selfcheck_avg(df, 'frontier'), 3)}",
        ]
    )
    rows.append(
        [
            "Benchmarks per model",
            str(int(_safe_get_metric(summary, "by_model", "oss") or 0)),
            str(int(_safe_get_metric(summary, "by_model", "frontier") or 0)),
        ]
    )

    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=colors.white,
        leading=9,
    )
    body: list[list[Any]] = []
    for r_idx, row in enumerate(rows):
        style_for_row = header_style if r_idx == 0 else styles["body"]
        body.append([Paragraph(value, style_for_row) for value in row])

    table = Table(body, colWidths=[1.95 * inch, 2.55 * inch, 2.55 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


# --- Charts side-by-side ---------------------------------------------------


def _chart_row(assets_dir: Path, styles: dict[str, ParagraphStyle]) -> Table:
    radar = assets_dir / "radar_chart.png"
    source_breakdown = assets_dir / "source_breakdown.png"
    chart_w = 3.25 * inch
    chart_h = 1.7 * inch

    if radar.exists():
        left: Any = Image(str(radar), width=chart_w, height=chart_h, kind="proportional")
    else:
        left = Paragraph(f"<i>Missing chart: {radar.name}</i>", styles["small"])
    if source_breakdown.exists():
        right: Any = Image(str(source_breakdown), width=chart_w, height=chart_h, kind="proportional")
    else:
        right = Paragraph(f"<i>Missing chart: {source_breakdown.name}</i>", styles["small"])

    table = Table([[left, right]], colWidths=[chart_w + 6, chart_w + 6], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


# --- Single page builder ---------------------------------------------------


def _build_story(
    summary: dict[str, Any],
    df: pd.DataFrame,
    assets_dir: Path,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = []
    story.append(Paragraph("AI Personal Assistant Benchmark", styles["title"]))
    story.append(
        Paragraph(
            "OSS path (Qwen2.5-0.5B with LLaMA-3.2-3B-free fallback) versus a frontier path "
            "(~openai/gpt-mini-latest), riding the same memory, tool, and guardrail core.",
            styles["subtitle"],
        )
    )

    story.append(
        Paragraph(
            "Two assistants share one core: token-budget memory, a small tool registry "
            "(web search, calculator, datetime), layered guardrails with toxicity scoring, "
            "Langfuse-ready tracing, and structured per-turn logs. Web search results are "
            "injected as a system message right before the user turn so the small OSS model "
            "stays grounded.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Evaluation uses three custom prompt banks (factual, adversarial, bias) inspired "
            "Evaluation now draws 100 rows each from TruthfulQA, ToxiGen, BOLD, "
            "RealToxicityPrompts, and JailbreakBench. A three-judge free-tier panel scores "
            "six dimensions, emits PASS / PARTIAL / FAIL by majority vote, and reports "
            "agreement so weak judge consensus is visible. A SelfCheckGPT-style NLI "
            "consistency check still runs on factual rows.",
            styles["body"],
        )
    )

    story.append(Paragraph("KPI comparison (live data)", styles["h2"]))
    story.append(_kpi_table(summary, df, styles))

    story.append(Spacer(1, 3))
    story.append(Paragraph("Infographics", styles["h2"]))
    story.append(_chart_row(assets_dir, styles))
    story.append(
        Paragraph(
            "Left: judge-score radar across six dimensions. "
            "Right: per-benchmark panel-score breakdown across the public suites.",
            styles["small"],
        )
    )

    story.append(Paragraph("Recommendations", styles["h2"]))
    story.extend(
        _bullet_list(
            [
                "Keep guardrails on by default. They materially improve jailbreak resistance "
                "at modest latency and near-zero actual free-tier spend on the OSS path.",
                "Watch panel agreement, not just the majority verdict. Low agreement is an "
                "early warning that the sample needs manual review.",
                "Use TruthfulQA + SelfCheck together for factuality: agreement without "
                "self-consistency is still a weak answer.",
                "Treat the equivalent-cost column as the budget planning number. Actual "
                "spend stays low because the panel leans on free-tier judges.",
                "Keep the public-benchmark source mix visible in every report so wins are "
                "not accidentally driven by one easy subset.",
            ],
            styles,
        )
    )

    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Repository: {GITHUB_URL} &nbsp;&nbsp; OSS Space: {HF_SPACE_URL}",
            styles["footer"],
        )
    )
    return story


# --- Entry point -----------------------------------------------------------


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(root / "eval" / "results")))
    assets_dir = root / "report" / "assets"
    out_pdf = root / "report" / "AI_Personal_Assistant_Benchmark_Report.pdf"

    summary = _load_summary(results_dir)
    df = _load_scores(results_dir)
    styles = _make_styles()

    page_w, page_h = letter
    left_margin = 0.45 * inch
    right_margin = 0.45 * inch
    top_margin = 0.4 * inch
    bottom_margin = 0.4 * inch
    frame_w = page_w - left_margin - right_margin
    frame_h = page_h - top_margin - bottom_margin

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="AI Personal Assistant Benchmark Report",
        author="Dual AI Assistant Benchmark",
    )

    story = _build_story(summary, df, assets_dir, styles)
    one_page = KeepInFrame(
        frame_w,
        frame_h,
        story,
        mode="shrink",
        hAlign="LEFT",
        vAlign="TOP",
    )
    doc.build([one_page])


if __name__ == "__main__":
    main()
