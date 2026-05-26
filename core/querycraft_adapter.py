import os
os.environ["USE_TF"]  = "0"
os.environ["USE_JAX"] = "0"

import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_ID    = "Sid9797/querycraft-phi3-sql"
SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language "
    "question, generate a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)

class QueryCraftAdapter:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self._loaded   = False

    def load(self):
        if self._loaded:
            return
        print("Loading QueryCraft model... (first load takes ~30 seconds)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        print("  Loading base model (Phi-3 Mini 3.8B)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_config,
            device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        print("  Loading QueryCraft adapter...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        self.model.eval()
        self._loaded = True
        print("✅ QueryCraft model ready")

    def generate(self, question: str, schema: str) -> dict:
        if not self._loaded:
            self.load()
        prompt = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question}\n\n"
            f"### SQL:\n"
        )
        inputs        = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = int((time.time() - start) * 1000)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        sql = self._clean_sql(raw_output)
        return {"sql": sql, "latency_ms": latency_ms, "prompt_tokens": prompt_tokens, "raw_output": raw_output}

    def _clean_sql(self, raw_output: str) -> str:
        if not raw_output or not raw_output.strip():
            return ""
        lines = raw_output.strip().split("\n")
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if sql_lines:
                    break
                continue
            sql_lines.append(stripped)
            if stripped.endswith(";"):
                break
        sql = " ".join(sql_lines).strip()
        sql = re.sub(r'"([^"]*)"', r"\'\1\'", sql)
        return sql