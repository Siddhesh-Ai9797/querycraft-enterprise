# Runs every QueryCraft benchmark on this PC, start to finish.
#
# Before you start:
#   1. Docker Desktop is open.
#   2. Your Chinook PostgreSQL database is running, and .env has DATABASE_URL and OPENAI_API_KEY.
#   3. conda activate ai-env
#   4. For the Modal step: modal setup, the querycraft-vllm secret, modal deploy done,
#      and MODAL_URL plus MODAL_API_KEY in .env.
#   5. Nothing else is using the GPU, such as a training run.
#
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_benchmarks.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_benchmarks.ps1 -N 200 -Skip modal
param(
  [int]$N = 500,
  [string[]]$Skip = @()
)
$ErrorActionPreference = "Stop"
$compose = "deploy/docker-compose.serving.yml"
$log = "benchmarks/results/run_$(Get-Date -Format yyyyMMdd-HHmm).log"
New-Item -ItemType Directory -Force benchmarks/results | Out-Null
Start-Transcript -Path $log | Out-Null

function Step($name) { Write-Host "`n==================== $name ====================" -ForegroundColor Cyan }

function Wait-Server($url, $seconds) {
  $deadline = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $deadline) {
    try { Invoke-RestMethod "$url/models" -TimeoutSec 5 | Out-Null; return } catch { Start-Sleep 5 }
  }
  throw "Server at $url did not start in $seconds seconds"
}

function Pipeline-Evals($backend) {
  $env:INFERENCE_BACKEND = $backend
  python evaluation/batch_evaluator.py
  python evaluation/ragas_evaluator.py
}

Step "Install packages"
pip install -q -r requirements.txt modal datasets

if (-not (Test-Path "models/querycraft-phi3-merged/config.json")) {
  Step "Merge adapter into Phi-3 Mini"
  python scripts/merge_adapter.py
}
if (-not (Test-Path "models/querycraft-phi3-q4_k_m.gguf")) {
  Step "Make 4-bit GGUF for llama.cpp"
  powershell -ExecutionPolicy Bypass -File deploy/make_gguf.ps1
}

if ($Skip -notcontains "local") {
  Step "Original setup: transformers, 4-bit, adapter on top"
  python benchmarks/sql_benchmark.py --backends local -n $N
  Pipeline-Evals "local"
}

if ($Skip -notcontains "openai") {
  Step "OpenAI baselines"
  python benchmarks/sql_benchmark.py --backends "gpt-3.5-turbo,gpt-4o-mini" -n $N
  $env:OPENAI_SQL_MODEL = "gpt-3.5-turbo"
  Pipeline-Evals "openai"
}

if ($Skip -notcontains "vllm") {
  Step "vLLM on this GPU"
  docker compose -f $compose up -d vllm
  Wait-Server "http://localhost:8001/v1" 600
  python benchmarks/sql_benchmark.py --backends vllm -n $N --concurrency 16 --throughput-n 200
  Pipeline-Evals "vllm"
  docker compose -f $compose stop vllm
}

if ($Skip -notcontains "llamacpp") {
  Step "llama.cpp on 2 CPU cores, like a t3.large"
  docker compose -f $compose up -d llamacpp
  Wait-Server "http://localhost:8002/v1" 300
  python benchmarks/sql_benchmark.py --backends llamacpp -n $N --concurrency 2 --throughput-n 40
  docker compose -f $compose stop llamacpp
}

if ($Skip -notcontains "modal") {
  Step "Modal L4 GPU, the live setup"
  python benchmarks/sql_benchmark.py --backends modal -n $N --concurrency 16 --throughput-n 200
  Pipeline-Evals "modal"
}

Step "Done"
Get-ChildItem benchmarks/results | Sort-Object LastWriteTime | Select-Object Name, LastWriteTime
Stop-Transcript | Out-Null
Write-Host "Send benchmarks\results to Claude, or just say done." -ForegroundColor Green
