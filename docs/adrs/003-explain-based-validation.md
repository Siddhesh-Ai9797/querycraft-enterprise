# ADR-003: PostgreSQL EXPLAIN for SQL Validation

**Date:** May 2026
**Status:** Accepted

## Context
Generated SQL must be validated before execution.
Options: sqlparse library, regex pattern matching, PostgreSQL EXPLAIN.

## Decision
Use PostgreSQL EXPLAIN as Layer 3, with keyword blocklist (Layer 2) and SELECT-only (Layer 1).

## Reasoning
- **sqlparse:** Cannot validate column/table existence or JOIN correctness
- **Regex:** Cannot handle SQL complexity, high false positive rate
- **PostgreSQL EXPLAIN:** Full query planner validation without executing or touching data

## Consequences
- Layer 3 validation requires a live database connection
- EXPLAIN adds ~5ms per validation call
- False negative rate effectively zero (same parser as execution)