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
# Run from the repo root, with -NoProfile so your conda env stays active:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_benchmarks.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_benchmarks.ps1 -N 200 -Skip modal
param(
  [int]$N = 500,
  [string[]]$Skip = @()
)
# Continue, not Stop: Windows PowerShell treats any line Python writes to stderr
# (even a warning) as an error once it is piped, which would end the run.
# Setup steps are checked with Must instead.
$ErrorActionPreference = "Continue"
# powershell -File passes "-Skip vllm,llamacpp" as one string, so split it here.
$Skip = @(($Skip -join ",").Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
# This project never uses TensorFlow; skip a broken install if one is present.
$env:USE_TF = "0"; $env:USE_JAX = "0"; $env:TRANSFORMERS_NO_TF = "1"
# Piped Python output falls back to the old Windows cp1252 encoding, which can't
# print the emoji in the code's status lines and crashes. Force UTF-8.
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
# Quiet two harmless warnings that otherwise flood the log.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:ANONYMIZED_TELEMETRY = "False"
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

# Stops the whole run if a setup step fails, instead of carrying on.
function Must($what) {
  if ($LASTEXITCODE -ne 0) { Stop-Transcript | Out-Null; throw "$what failed, stopping." }
}

# Benchmarks keep going if one fails, so one broken backend doesn't lose the rest.
function Pipeline-Evals($backend) {
  $env:INFERENCE_BACKEND = $backend
  python evaluation/batch_evaluator.py 2>&1 | Out-Host
  if ($LASTEXITCODE -ne 0) { Write-Host "Pipeline test failed for $backend, continuing" -ForegroundColor Yellow }
  python evaluation/ragas_evaluator.py 2>&1 | Out-Host
  if ($LASTEXITCODE -ne 0) { Write-Host "RAGAS failed for $backend, continuing" -ForegroundColor Yellow }
}

Step "Check Python environment"
python -c "import sys; print('Python:', sys.executable)"
python -c "import torch, transformers, peft, mlflow, sqlalchemy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Must "Finding torch, transformers, peft, mlflow and sqlalchemy. Run conda activate ai-env and use -NoProfile"
# Only the two packages this benchmark adds. Your env already has the rest.
pip install -q datasets modal
Must "Installing datasets and modal"

if (-not (Test-Path "models/querycraft-phi3-merged/config.json")) {
  Step "Merge adapter into Phi-3 Mini"
  python scripts/merge_adapter.py
  Must "Merging the adapter"
}
if (-not (Test-Path "models/querycraft-phi3-q4_k_m.gguf")) {
  Step "Make 4-bit GGUF for llama.cpp"
  powershell -NoProfile -ExecutionPolicy Bypass -File deploy/make_gguf.ps1
  Must "Making the GGUF file"
}

if ($Skip -notcontains "local") {
  Step "Original setup: transformers, 4-bit, adapter on top"
  python benchmarks/sql_benchmark.py --backends local -n $N 2>&1 | Out-Host
  Pipeline-Evals "local"
}

if ($Skip -notcontains "openai") {
  Step "OpenAI baselines"
  python benchmarks/sql_benchmark.py --backends gpt-4o-mini -n $N 2>&1 | Out-Host
  $env:OPENAI_SQL_MODEL = "gpt-4o-mini"
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
