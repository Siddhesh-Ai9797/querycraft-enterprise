# ADR-005: LangChain LCEL over LangGraph

**Date:** May 2026
**Status:** Accepted

## Context
The pipeline requires orchestrating 5 agents in sequence with one conditional retry loop.
Options: LangChain LCEL, LangGraph, plain Python.

## Decision
Use LangChain LCEL (RunnableLambda) for the linear chain, plain Python for the retry loop.

## Reasoning
- **LangGraph:** Designed for stateful, cyclical, graph-based workflows — overkill here
- **Pure Python:** No composability, no built-in tracing
- **LCEL:** Correct tool for linear chains, composable, swappable agents

## Consequences
- Each agent wrapped as RunnableLambda
- Retry loop (max 2 attempts) in plain Python inside pipeline.run()
- Swapping any agent requires changing one line in pipeline.__init__