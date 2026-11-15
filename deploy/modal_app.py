"""
Live serving for the fine tuned QueryCraft model on Modal.

vLLM runs the merged model on an L4 GPU behind an OpenAI compatible API.
The GPU scales to zero when idle, so you only pay while requests run.

One time setup, from the repo root:
    pip install modal
    modal setup
    modal secret create querycraft-vllm VLLM_API_KEY=<a long random string>
    modal run deploy/modal_app.py::merge      # builds the merged model once
    modal deploy deploy/modal_app.py          # prints the server URL

Then point the API at it:
    INFERENCE_BACKEND=modal
    MODAL_URL=https://<your-workspace>--querycraft-sql-serve.modal.run/v1
    MODAL_API_KEY=<the same random string>
"""
import os
import subprocess

import modal

APP_NAME    = "querycraft-sql"
MODEL_DIR   = "/models/querycraft-phi3-merged"
SERVED_NAME = "querycraft-sql"
GPU         = os.getenv("QUERYCRAFT_GPU", "L4")
VLLM_PORT   = 8000

app    = modal.App(APP_NAME)
volume = modal.Volume.from_name("querycraft-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("querycraft-hf-cache", create_if_missing=True)

merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "transformers==4.44.2", "peft==0.12.0",
                 "accelerate==0.34.2", "safetensors")
    .add_local_file("scripts/merge_adapter.py", "/root/merge_adapter.py")
)

# vLLM's own image, so every library is the exact version vLLM was tested with.
# Installing vllm with pip pulled in transformers 5 and openai 3, which this
# vLLM release was never tested against.
vllm_image = modal.Image.from_registry("vllm/vllm-openai:v0.10.2").entrypoint([])


@app.function(image=merge_image, volumes={"/models": volume, "/root/.cache/huggingface": hf_cache},
              cpu=4, memory=32768, timeout=3600)
def merge():
    """Merges the LoRA adapter into Phi-3 Mini once and stores it in a Modal volume."""
    import sys
    sys.path.insert(0, "/root")
    from merge_adapter import merge as do_merge
    do_merge(MODEL_DIR, device="cpu")
    volume.commit()


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={"/models": volume, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("querycraft-vllm")],
    scaledown_window=300,   # stay warm 5 minutes after the last request
    timeout=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    subprocess.Popen([
        "vllm", "serve", MODEL_DIR,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
    ])
