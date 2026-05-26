# ADR-004: GPT-3.5-turbo for Answer Synthesis

**Date:** May 2026
**Status:** Accepted

## Context
The Answer Synthesizer converts raw SQL results into natural language answers.
Options: GPT-4o, GPT-3.5-turbo, local Phi-3 Mini, template-based formatting.

## Decision
Use GPT-3.5-turbo for answer synthesis.

## Reasoning
Converting {"count": 275} to "There are 275 artists" does not require GPT-4 reasoning depth.
RAGAS evaluation confirmed GPT-3.5-turbo answer relevancy (0.976) matches GPT-4o (0.941)
for this specific task at 5x lower cost.

## Consequences
- Answer synthesis costs ~$0.0004 per query
- Temperature set to 0.1 for consistent, factual responses