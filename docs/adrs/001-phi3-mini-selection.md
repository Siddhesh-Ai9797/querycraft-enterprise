# ADR-001: Phi-3 Mini as SQL Generation Model

**Date:** May 2026
**Status:** Accepted

## Context
QueryCraft Enterprise needs a model to generate SQL from natural language.
Options considered: GPT-4, GPT-3.5-turbo, LLaMA 3 8B, Mistral 7B, Phi-3 Mini 3.8B.

## Decision
Use a fine-tuned Phi-3 Mini 3.8B as the SQL generation component.

## Reasoning

| Model | GPU Memory | Fine-tunable | Cost/Query | SQL Accuracy |
|---|---|---|---|---|
| GPT-4 | N/A (API) | No | ~$0.03 | High |
| LLaMA 3 8B | ~6GB (4-bit) | Yes | $0 | Medium |
| Phi-3 Mini 3.8B | ~2.5GB (4-bit) | Yes | $0 | High after fine-tuning |

Phi-3 Mini fits on consumer GPU, trains in ~3 hours, and achieves 82% exact match
accuracy after fine-tuning on 76,577 examples — compared to 0% for the base model.

## Consequences
- SQL generation has zero marginal API cost in local deployment
- Cloud deployment switches to GPT-3.5-turbo via INFERENCE_BACKEND env var
- Model accuracy bounded by training data complexity (struggles with 4+ table joins)