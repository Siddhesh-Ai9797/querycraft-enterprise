import os
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language "
    "question, generate a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)

class ProductionQueryAdapter:
    def __init__(self):
        self.client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._loaded = True

    def load(self):
        print("✅ Production adapter ready (GPT-3.5-turbo SQL generation)")

    def generate(self, question: str, schema: str) -> dict:
        prompt = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Schema:\n{schema}\n\n"
            f"### Question:\n{question}\n\n"
            f"### SQL:\n"
        )
        start = time.time()
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=150,
        )
        latency_ms = int((time.time() - start) * 1000)
        raw_sql = response.choices[0].message.content.strip()
        sql = raw_sql.replace("```sql", "").replace("```", "").strip()
        sql = re.sub(r'"([^"]*)"', r"\'\1\'", sql)
        return {
            "sql": sql, "latency_ms": latency_ms,
            "prompt_tokens": response.usage.prompt_tokens, "raw_output": raw_sql,
        }