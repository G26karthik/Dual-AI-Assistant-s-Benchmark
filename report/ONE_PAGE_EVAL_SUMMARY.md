# One-Page Evaluation Summary Content

Use this content directly for the final one-page assignment PDF if you want to re-layout it in any external design tool.

## Title
AI Personal Assistant Benchmark Report

## Models
- OSS: Qwen2.5-0.5B-Instruct (Hugging Face Space)
- Frontier: OpenRouter `~openai/gpt-mini-latest`
- Safety layer: Llama Guard 3

## Methodology
- Prompt set: factual, adversarial/jailbreak, and bias/sensitive prompts
- Evaluation dimensions: hallucination resistance, safety/refusal behavior, bias handling, helpfulness
- Judge: LLM-as-judge pipeline (`eval/judge.py`)
- Hallucination support signal: SelfCheckGPT-style consistency checks for factual prompts

## Comparison Snapshot (latest full run)
- Rows evaluated: 88 total (44 OSS + 44 Frontier)
- Avg latency (ms): OSS 5985.05 | Frontier 6098.84
- P50 latency (ms): OSS 5376.00 | Frontier 6097.50
- P95 latency (ms): OSS 11327.45 | Frontier 7266.05
- Estimated eval cost (USD): OSS 0.00000 | Frontier 0.02417

## Key Findings
- OSS path provides a no-cost serving baseline and acceptable guardrail behavior for constrained tasks.
- Frontier path improves response consistency and routing reliability for benchmark execution.
- Guardrails and explicit refusal behavior are critical for adversarial prompts in both stacks.

## Recommendations
- Keep guardrails enabled by default and log all block events.
- Use frontier path for higher-stakes responses; use OSS for low-cost/throughput flows.
- Add periodic benchmark regression runs to detect quality or safety drift.
- Expand multi-turn adversarial coverage beyond A12 to strengthen robustness claims.

## Visual Assets to include
- `report/assets/radar_chart.png`
- `report/assets/latency_cost_bar.png`
- (optional if space permits) `report/assets/pass_fail_summary.png`
