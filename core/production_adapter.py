import os
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from core.sql_text import build_prompt, clean_sql

OPENAI_SQL_MODEL = os.getenv("OPENAI_SQL_MODEL", "gpt-4o-mini")

class ProductionQueryAdapter:
    def __init__(self):
        self.client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._loaded = True

    def load(self):
        print(f"✅ Production adapter ready ({OPENAI_SQL_MODEL} SQL generation)")

    def generate(self, question: str, schema: str) -> dict:
        prompt = build_prompt(question, schema)
        start = time.time()
        response = self.client.chat.completions.create(
            model=OPENAI_SQL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=150,
        )
        latency_ms = int((time.time() - start) * 1000)
        raw_sql = response.choices[0].message.content.strip()
        sql = clean_sql(raw_sql)
        return {
            "sql": sql, "latency_ms": latency_ms,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens, "raw_output": raw_sql,
        }