import torch
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from core.pipeline import QueryCraftPipeline
load_dotenv()

pipeline: Optional[QueryCraftPipeline] = None
_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Starting QueryCraft Enterprise API...")
    pipeline = QueryCraftPipeline()
    pipeline.initialize()
    print("✅ API ready to serve requests")
    yield
    print("Shutting down...")

app = FastAPI(title="QueryCraft Enterprise API", version="1.0.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500,
                          example="How many artists are there?")

class QueryResponse(BaseModel):
    question:       str
    answer:         str
    sql:            Optional[str]   = None
    confidence:     float           = 0.0
    quality_score:  Optional[float] = None
    sql_complexity: Optional[str]   = None
    tables_used:    list            = []
    total_ms:       int             = 0
    attempts:       int             = 1
    success:        bool            = False
    error:          Optional[str]   = None
    flagged:        Optional[bool]  = None

class HealthResponse(BaseModel):
    status: str; model: str; version: str; uptime_sec: float

@app.get("/")
def root():
    return {"service": "QueryCraft Enterprise API", "version": "1.0.0",
            "endpoints": {"query": "POST /query", "health": "GET /health", "docs": "GET /docs"}}

@app.get("/health", response_model=HealthResponse)
def health_check():
    if pipeline is None or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return HealthResponse(status="healthy", model="querycraft-phi3-sql",
                          version="1.0.0", uptime_sec=round(time.time()-_start_time, 1))

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    try:
        result = pipeline.run(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)[:200]}")

@app.get("/metrics")
def metrics():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri("./mlruns")
        client     = MlflowClient()
        experiment = client.get_experiment_by_name("querycraft-enterprise")
        if not experiment: return {"error": "No runs found"}
        runs       = client.search_runs([experiment.experiment_id], max_results=100)
        total      = len(runs)
        successful = sum(1 for r in runs if r.data.metrics.get("success",0)==1)
        quality    = [r.data.metrics["quality_score"] for r in runs if "quality_score" in r.data.metrics]
        latencies  = [r.data.metrics["total_ms"] for r in runs if "total_ms" in r.data.metrics]
        return {"total_queries": total,
                "success_rate": round(successful/total,3) if total else 0,
                "avg_quality": round(sum(quality)/len(quality),3) if quality else None,
                "avg_latency_ms": round(sum(latencies)/len(latencies),0) if latencies else None}
    except Exception as e:
        return {"error": str(e)[:200]}