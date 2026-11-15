# Turns the merged model into a 4-bit GGUF file for llama.cpp. Run from the repo root.
if (-not (Test-Path "models/querycraft-phi3-merged/config.json")) {
  Write-Host "Run python scripts/merge_adapter.py first" -ForegroundColor Red; exit 1
}
$models = (Resolve-Path "models").Path
docker run --rm -v "${models}:/models" ghcr.io/ggml-org/llama.cpp:full `
  --convert "/models/querycraft-phi3-merged" --outtype f16 --outfile /models/querycraft-phi3-f16.gguf
if ($LASTEXITCODE -ne 0) { exit 1 }
docker run --rm -v "${models}:/models" ghcr.io/ggml-org/llama.cpp:full `
  --quantize /models/querycraft-phi3-f16.gguf /models/querycraft-phi3-q4_k_m.gguf Q4_K_M
if ($LASTEXITCODE -ne 0) { exit 1 }
# The f16 file is only a stepping stone; the 4-bit file is what gets served.
Remove-Item "$models\querycraft-phi3-f16.gguf" -ErrorAction SilentlyContinue
Get-Item "$models\querycraft-phi3-q4_k_m.gguf" | Select-Object Name, @{n="GB";e={[math]::Round($_.Length/1GB,2)}}
