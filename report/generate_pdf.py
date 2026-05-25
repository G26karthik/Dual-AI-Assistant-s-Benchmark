from __future__ import annotations

import json
import os
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _draw_key_value(c: canvas.Canvas, x: int, y: int, key: str, value: str) -> int:
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, f"{key}:")
    c.setFont("Helvetica", 10)
    c.drawString(x + 95, y, value)
    return y - 14


def _draw_image_if_exists(c: canvas.Canvas, image_path: Path, x: int, y: int, max_w: int, max_h: int) -> int:
    if not image_path.exists():
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(x, y, f"Missing chart: {image_path.name}")
        return y - 16
    img = ImageReader(str(image_path))
    src_w, src_h = img.getSize()
    scale = min(max_w / src_w, max_h / src_h)
    draw_w = src_w * scale
    draw_h = src_h * scale
    c.drawImage(img, x, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
    return int(y - draw_h - 12)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", str(root / "eval" / "results")))
    assets_dir = root / "report" / "assets"
    out_pdf = root / "report" / "AI_Personal_Assistant_Benchmark_Report.pdf"
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter
    y = height - 40

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "AI Personal Assistant Benchmark Report")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(40, y, "Models: OSS (Qwen2.5 OSS track) vs Frontier (OpenAI GPT-4.1 track)")
    y -= 14
    c.drawString(40, y, "Prompt mix: factual + adversarial/jailbreak + sensitive/bias")
    y -= 14
    c.drawString(40, y, f"Artifacts: {output_dir}")
    y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Deployment Cost + Latency Snapshot")
    y -= 16

    avg_latency = summary.get("avg_latency_ms", {})
    p50_latency = summary.get("p50_latency_ms", {})
    p95_latency = summary.get("p95_latency_ms", {})
    est_cost = summary.get("estimated_total_cost_usd", {})

    y = _draw_key_value(c, 40, y, "OSS avg/p50/p95 ms", f"{avg_latency.get('oss', 0):.1f} / {p50_latency.get('oss', 0):.1f} / {p95_latency.get('oss', 0):.1f}")
    y = _draw_key_value(
        c,
        40,
        y,
        "Frontier avg/p50/p95 ms",
        f"{avg_latency.get('frontier', 0):.1f} / {p50_latency.get('frontier', 0):.1f} / {p95_latency.get('frontier', 0):.1f}",
    )
    y = _draw_key_value(c, 40, y, "OSS eval cost (USD)", f"{est_cost.get('oss', 0):.4f}")
    y = _draw_key_value(c, 40, y, "Frontier eval cost (USD est.)", f"{est_cost.get('frontier', 0):.4f}")
    y -= 10

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Infographics")
    y -= 14

    chart_paths = [
        assets_dir / "radar_chart.png",
        assets_dir / "grouped_bar.png",
        assets_dir / "safety_heatmap.png",
        assets_dir / "pass_fail_summary.png",
    ]
    for image_path in chart_paths:
        y = _draw_image_if_exists(c, image_path, x=40, y=y, max_w=int(width - 80), max_h=145)
        if y < 120 and image_path != chart_paths[-1]:
            c.showPage()
            y = height - 40

    if y < 130:
        c.showPage()
        y = height - 40
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Recommendations")
    y -= 16
    c.setFont("Helvetica", 10)
    recommendations = [
        "- Keep OSS guardrails enabled by default; they materially improve jailbreak resistance.",
        "- Use web search grounding for uncertain factual prompts to reduce hallucinations.",
        "- Favor frontier model for high-stakes responses; use OSS for low-cost automation paths.",
        "- Add periodic benchmark regression runs in CI to catch safety/quality drift.",
    ]
    for line in recommendations:
        c.drawString(40, y, line)
        y -= 14

    c.save()


if __name__ == "__main__":
    main()
