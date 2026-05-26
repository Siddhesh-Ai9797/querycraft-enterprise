# QueryCraft Enterprise

Production multi-agent Text-to-SQL platform built on a fine-tuned Phi-3 Mini model.
Converts natural language questions about a music store database into accurate answers
with full validation, monitoring, and cloud deployment.

**Live Demo:** http://ec2-13-220-173-253.compute-1.amazonaws.com:8000/docs

---

## What It Does

Ask a question in plain English. Get a verified, confident answer backed by real data.
POST /query
{"question": "Which artist has the most albums?"}
→ {"answer": "Iron Maiden has the most albums.", "confidence": 1.0, "sql": "SELECT..."}

---

## Architecture
User Question
↓
Schema Discovery Agent     — Chroma vector search selects relevant tables
↓
QueryCraft SQL Generator   — Fine-tuned Phi-3 Mini (82% exact match accuracy)
↓
SQL Validator Agent        — 3-layer safety: SELECT-only + keyword blocklist + EXPLAIN
↓
SQL Executor               — Runs against live PostgreSQL (Chinook database)
↓
Answer Synthesizer         — GPT-3.5-turbo converts results to natural language
↓
Quality Monitor            — Composite score logged to MLflow
↓
User Answer + Confidence Score

---

## Stack

| Layer | Technology |
|---|---|
| SQL Generation | Fine-tuned Phi-3 Mini 3.8B (QLoRA) |
| Agent Orchestration | LangChain LCEL |
| Schema Cache | Chroma vector database |
| Evaluation | RAGAS |
| Experiment Tracking | MLflow (63+ runs logged) |
| API | FastAPI + Uvicorn |
| Monitoring | Streamlit dashboard |
| Deployment | Docker → AWS ECR → EC2 |
| Load Testing | Locust |

---

## Evaluation Results

### RAGAS Evaluation (30 stratified queries)

| System | $/1K queries | Answer Relevancy | Faithfulness |
|---|---|---|---|
| **QueryCraft Enterprise** | **$0.40** | **0.971** | 0.071 |
| GPT-3.5-turbo baseline | $1.07 | 0.976 | 0.036 |
| GPT-4o baseline | $5.32 | 0.941 | 0.033 |

**QueryCraft Enterprise achieves GPT-3.5-turbo quality at 2.7x lower cost.**

### AWS Load Test (Locust, 5 concurrent users, 3 minutes)

| Metric | Value |
|---|---|
| p50 latency | 1200ms |
| p95 latency | 2000ms |
| p99 latency | 7500ms |
| Total requests | 374 |
| Failure rate | 0% |

---

## Quick Start

```bash
git clone https://github.com/Siddhesh-Ai9797/querycraft-enterprise
cd querycraft-enterprise
conda create -n qce-env python=3.11
conda activate qce-env
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Related Projects

- **[HuggingFace Model](https://huggingface.co/Sid9797/querycraft-phi3-sql)** — Published adapter weights