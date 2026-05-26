import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """You are a data analyst assistant. You are given a question, SQL, and results.
Write a clear concise answer. Respond with valid JSON only.
Format: {"answer": "...", "confidence": 0.95}
Confidence: 0.95+ = complete, 0.80+ = partial, 0.30+ = empty/wrong"""

class AnswerSynthesizerAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = "gpt-3.5-turbo"

    def synthesize(self, question: str, sql: str, execution_result: dict) -> dict:
        if not execution_result["success"]:
            return {"answer": "Unable to retrieve results.", "confidence": 0.1,
                    "latency_ms": 0, "tokens_used": 0, "error": execution_result["error"]}
        if execution_result["row_count"] == 0:
            return {"answer": "No results were found for that question.",
                    "confidence": 0.3, "latency_ms": 0, "tokens_used": 0, "error": None}
        rows    = execution_result["rows"][:10]
        columns = execution_result["columns"]
        rc      = execution_result["row_count"]
        note    = f"\n(Showing 10 of {rc} rows)" if rc > 10 else ""
        user_msg = f"Question: {question}\n\nSQL: {sql}\n\nColumns: {columns}\nResults: {rows}{note}"
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_msg}],
                temperature=0.1, max_tokens=200, response_format={"type":"json_object"},
            )
            latency_ms  = int((time.time()-start)*1000)
            parsed      = json.loads(response.choices[0].message.content)
            return {"answer": parsed.get("answer","Unable to generate answer."),
                    "confidence": float(parsed.get("confidence",0.5)),
                    "latency_ms": latency_ms, "tokens_used": response.usage.total_tokens, "error": None}
        except Exception as e:
            return {"answer": "Error generating answer.", "confidence": 0.1,
                    "latency_ms": 0, "tokens_used": 0, "error": str(e)[:200]}