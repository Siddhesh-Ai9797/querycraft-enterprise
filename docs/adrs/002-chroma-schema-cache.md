# ADR-002: Chroma for Schema Vector Cache

**Date:** May 2026
**Status:** Accepted

## Context
The pipeline needs to select 2-3 relevant tables from 11 Chinook tables per query.
Options: query PostgreSQL every time, FAISS in-memory index, Chroma persistent vector database.

## Decision
Use Chroma with SentenceTransformer embeddings (all-MiniLM-L6-v2).

## Reasoning
- **PostgreSQL introspection every query:** ~80ms overhead, no semantic matching
- **FAISS:** In-memory only, no persistence across restarts
- **Chroma:** Persistent disk storage, semantic similarity search, 5ms query time

## Consequences
- Schema cache built once on first startup (~2 seconds)
- Subsequent queries: 5ms schema retrieval vs 80ms introspection
- Cache invalidation required on schema changes