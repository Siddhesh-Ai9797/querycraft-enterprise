# Turns the merged model into a 4-bit GGUF file for llama.cpp. Run from the repo root.
$models = (Resolve-Path "models").Path
docker run --rm -v "${models}:/models" ghcr.io/ggml-org/llama.cpp:full `
  --convert "/models/querycraft-phi3-merged" --outtype f16 --outfile /models/querycraft-phi3-f16.gguf
docker run --rm -v "${models}:/models" ghcr.io/ggml-org/llama.cpp:full `
  --quantize /models/querycraft-phi3-f16.gguf /models/querycraft-phi3-q4_k_m.gguf Q4_K_M
Get-Item "$models\querycraft-phi3-q4_k_m.gguf" | Select-Object Name, @{n="GB";e={[math]::Round($_.Length/1GB,2)}}
