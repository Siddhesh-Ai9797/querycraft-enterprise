"""
Runs the fine tuned QueryCraft model behind any OpenAI compatible server.

The same adapter talks to all three ways of serving the model:

    vllm      vLLM on a GPU, locally in Docker
    modal     vLLM on a Modal GPU that scales to zero (the live demo)
    llamacpp  llama.cpp running a 4-bit GGUF file on a CPU

It sends the exact raw prompt the model was fine tuned on to the
/v1/completions endpoint, not a chat template, so answers match training.
"""
import os
import time

from openai import OpenAI

from core.sql_text import STOP_SEQUENCES, build_prompt, clean_sql

DEFAULTS = {
    "vllm"    : {"url": "http://localhost:8001/v1", "model": "querycraft-sql"},
    "llamacpp": {"url": "http://localhost:8002/v1", "model": "querycraft-sql"},
    "modal"   : {"url": None,                        "model": "querycraft-sql"},
}


class RemoteSQLAdapter:
    def __init__(self, backend: str, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float | None = None):
        if backend not in DEFAULTS:
            raise ValueError(f"Unknown backend {backend}")
        self.backend  = backend
        prefix = backend.upper()   # VLLM_URL, MODAL_URL, LLAMACPP_URL
        self.base_url = (base_url or os.getenv(f"{prefix}_URL") or os.getenv("SQL_MODEL_URL")
                         or DEFAULTS[backend]["url"])
        self.model    = model or os.getenv("SQL_MODEL_NAME") or DEFAULTS[backend]["model"]
        if not self.base_url:
            raise ValueError(f"Set {prefix}_URL for the {backend} backend")
        # Modal containers can take a minute to wake up from zero.
        default_timeout = 180.0 if backend == "modal" else 60.0
        self.client = OpenAI(
            base_url = self.base_url,
            api_key  = (api_key or os.getenv(f"{prefix}_API_KEY") or os.getenv("SQL_MODEL_API_KEY")
                        or "not-needed"),
            timeout  = timeout or float(os.getenv("SQL_MODEL_TIMEOUT", default_timeout)),
            max_retries = 1,
        )
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            models = [m.id for m in self.client.models.list().data]
        except Exception as e:  # noqa: BLE001
            # A sleeping Modal GPU or a slow network must not stop the app from
            # starting. The first question wakes the server instead.
            print(f"⚠️  {self.backend} server not reachable yet ({type(e).__name__}); will retry on first question")
            return
        if self.model not in models:
            raise RuntimeError(f"{self.base_url} serves {models}, not {self.model}")
        self._loaded = True
        print(f"✅ Fine tuned model ready on {self.backend} at {self.base_url}")

    def generate(self, question: str, schema: str) -> dict:
        prompt = build_prompt(question, schema)
        start = time.time()
        response = self.client.completions.create(
            model       = self.model,
            prompt      = prompt,
            max_tokens  = 128,
            temperature = 0,
            stop        = STOP_SEQUENCES,
        )
        latency_ms = int((time.time() - start) * 1000)
        raw = response.choices[0].text or ""
        usage = response.usage
        return {
            "sql"              : clean_sql(raw),
            "latency_ms"       : latency_ms,
            "prompt_tokens"    : getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "raw_output"       : raw,
        }
