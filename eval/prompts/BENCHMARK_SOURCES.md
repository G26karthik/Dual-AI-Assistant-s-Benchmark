# Benchmark Source Mapping

This project now loads public benchmark datasets directly and samples them down
to a lean reproducible suite.

## Factual / Hallucination
- Dataset: `truthfulqa/truthful_qa` (`multiple_choice`, `validation` split).
- Public fallback: `EleutherAI/truthful_qa_mc` if the primary dataset is unavailable.
- Coverage: factual accuracy, misconception handling, and hallucination resistance.

## Adversarial / Jailbreak / Safety
- Dataset: `JailbreakBench/JBB-Behaviors` (`behaviors`, `harmful` split).
- Coverage: jailbreak refusal, harmful-behavior refusal, and safety-policy robustness.

## Bias
- Dataset: `AmazonScience/bold` (`train` split).
- Coverage: bias-sensitive prompt continuation across gender, religion, race, politics, and professions.

## Toxicity / Harmful Output
- Dataset: `toxigen/toxigen-data` (`annotated`, `train` split).
- Public mirror fallback: `concretejungles/toxigen-paraphrased` when the official ToxiGen repo is gated or unavailable.
- Dataset: `allenai/real-toxicity-prompts` (`train` split).
- Coverage: hateful/toxic prompt handling plus toxic continuation resistance on real web text.

## Sampling policy
- Default sample size: `100` rows per benchmark.
- Sampling is deterministic and controlled by per-benchmark env seeds.
- The benchmark loader keeps the exact source dataset id on every row, so the
  analysis and report can break results down by benchmark origin.
