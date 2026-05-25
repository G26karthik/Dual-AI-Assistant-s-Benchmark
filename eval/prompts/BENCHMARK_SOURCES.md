# Benchmark Source Mapping

This project uses custom prompt sets inspired by public benchmark families so that runs stay lightweight while preserving benchmark intent.

## Factual / Hallucination
- Primary inspiration: **TruthfulQA**, **SimpleQA-style factual checks**, and common factuality probes.
- Adaptation approach: short factual prompts, false-premise traps, and uncertainty-sensitive questions.

## Adversarial / Jailbreak / Safety
- Primary inspiration: **AdvBench**, jailbreak corpora, and prompt-injection attack patterns.
- Adaptation approach: direct jailbreak attempts, role-play bypasses, system prompt extraction, and multi-turn escalation.

## Bias / Harmful Output
- Primary inspiration: **BBQ** (Bias Benchmark for QA), **BOLD** style sensitive prompts, and stereotype-risk prompts.
- Adaptation approach: stereotype probing across gender, race, religion, disability, socioeconomic status, and nationality.

## Why this adaptation
- Keeps evaluation runnable in small projects without heavy benchmark harness dependencies.
- Preserves the assignment-required axes:
  - Hallucination resistance
  - Bias and harmful outputs
  - Content safety and jailbreak robustness
