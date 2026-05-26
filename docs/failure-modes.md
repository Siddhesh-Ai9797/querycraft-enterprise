# QueryCraft Enterprise — Failure Mode Analysis

## Failure Mode 1: Multi-Table Revenue Queries

**What fails:** Questions requiring 4+ table JOINs for revenue calculations.
**Root cause:** Training data predominantly single/two-table examples.
**Observed rate:** 3/20 (15%) in batch evaluation, all hard-difficulty queries.
**Mitigation:** Pre-computed revenue view in PostgreSQL; retry with error feedback corrects ~40%.

---

## Failure Mode 2: Double-Quote String Literals

**What fails:** Model generates WHERE name = "Iron Maiden" instead of WHERE name = 'Iron Maiden'
**Root cause:** Training data mixed SQL dialects (MySQL/SQLite accept double quotes; PostgreSQL does not).
**Mitigation:** _clean_sql() regex substitution in QueryCraftAdapter. **Fully mitigated.**

---

## Failure Mode 3: Stale Schema Cache

**What fails:** Schema Discovery returns columns that no longer exist after migration.
**Root cause:** Chroma PersistentClient caches schema at first startup.
**Mitigation:** SchemaDiscoveryAgent.refresh() rebuilds cache. Manual — automated invalidation is v2 enhancement.

---

## Failure Mode 4: OpenAI Rate Limiting Under Load

**What fails:** Answer Synthesizer returns 429 under sustained concurrent load.
**Observed:** p99 latency spike to 7.5s during Locust test (5 concurrent users).
**Mitigation:** Upgrade to tier-2 OpenAI account; exponential backoff in AnswerSynthesizerAgent.

---

## Failure Mode 5: ONNX Runtime DLL Conflict on Windows

**What fails:** Application crashes on import — OSError: DLL load failed while importing c10.dll
**Root cause:** onnxruntime and torch both initialize CUDA DLLs; load order conflict on Windows.
**Mitigation:** `import torch` is the absolute first import in core/pipeline.py. **Fully mitigated.**