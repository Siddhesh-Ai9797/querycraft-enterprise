import os
# This project never uses TensorFlow or JAX. Telling transformers so avoids
# loading a broken TensorFlow install that may sit in the same environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# torch must be imported before onnxruntime on Windows, or CUDA DLLs clash.
try:
    import torch  # noqa: F401
except ImportError:
    pass
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
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
    if os.getenv("SEED_CHINOOK", "0") == "1":
        from scripts.seed_chinook import seed
        seed()
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
    columns:        list            = []
    rows:           list            = []
    row_count:      int             = 0
    steps:          dict            = {}
    sql_backend:    Optional[str]   = None
    model:          Optional[str]   = None

class HealthResponse(BaseModel):
    status: str; model: str; sql_backend: str; version: str; uptime_sec: float

STATIC_DIR = Path(__file__).parent / "static"

# ── Abuse limits for the public demo ─────────────────────────────────────
# Every question costs a Modal GPU call and an OpenAI call, so cap how many
# one visitor can ask per minute and how many the whole demo serves per day.
PER_IP_PER_MINUTE = int(os.getenv("PER_IP_PER_MINUTE", "6"))
DAILY_LIMIT       = int(os.getenv("DAILY_QUERY_LIMIT", "300"))
_ip_calls: dict = defaultdict(deque)
_daily = {"day": time.strftime("%Y-%m-%d"), "count": 0}
_limit_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


def _check_limits(request: Request):
    now = time.time()
    with _limit_lock:
        today = time.strftime("%Y-%m-%d")
        if _daily["day"] != today:
            _daily.update(day=today, count=0)
        if _daily["count"] >= DAILY_LIMIT:
            raise HTTPException(status_code=429, detail="The demo has answered its daily limit of questions. Try again tomorrow.")
        calls = _ip_calls[_client_ip(request)]
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= PER_IP_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many questions in a minute. Wait a moment and ask again.")
        calls.append(now)
        _daily["count"] += 1


def _model_label(backend: str) -> str:
    if backend == "openai":
        return f"{os.getenv('OPENAI_SQL_MODEL', 'gpt-4o-mini')} baseline"
    where = {"modal": "vLLM on a Modal L4 GPU", "vllm": "vLLM", "llamacpp": "llama.cpp, 4-bit on CPU",
             "local": "local GPU, 4-bit"}.get(backend, backend)
    return f"Fine tuned Phi-3 Mini 3.8B, {where}"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_info():
    return {"service": "QueryCraft Enterprise API", "version": "1.0.0",
            "endpoints": {"query": "POST /query", "health": "GET /health", "docs": "GET /docs"}}

@app.get("/health", response_model=HealthResponse)
def health_check():
    if pipeline is None or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    backend = pipeline.inference_backend
    return HealthResponse(status="healthy", model=_model_label(backend), sql_backend=backend,
                          version="1.0.0", uptime_sec=round(time.time()-_start_time, 1))

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
    if pipeline is None or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    _check_limits(http_request)
    try:
        result = pipeline.run(request.question)
        result.pop("schema_context", None)
        result["sql_backend"] = pipeline.inference_backend
        result["model"] = _model_label(pipeline.inference_backend)
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