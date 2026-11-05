"""
Decides which test files run based on what is available.

Unit tests always run. Files that need Postgres run when DATABASE_URL is set,
files that call OpenAI run when OPENAI_API_KEY is set, and files that build the
Chroma schema cache run when chromadb is installed. CI provides a Chinook
Postgres database, so everything except the OpenAI tests runs there.
"""
import importlib.util
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

NEEDS_DB = {
    "test_db_connection.py", "test_chroma.py", "test_schema_discovery.py", "test_sql_executor.py",
    "test_sql_validator.py", "test_pipeline.py", "test_querycraft_adapter.py",
}
NEEDS_OPENAI = {"test_answer_synthesizer.py", "test_pipeline.py", "test_querycraft_adapter.py"}
NEEDS_CHROMA = {"test_chroma.py", "test_schema_discovery.py", "test_pipeline.py"}

collect_ignore = []
if not os.getenv("DATABASE_URL"):
    collect_ignore += sorted(NEEDS_DB)
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-your"):
    collect_ignore += sorted(NEEDS_OPENAI)
if importlib.util.find_spec("chromadb") is None:
    collect_ignore += sorted(NEEDS_CHROMA)
collect_ignore = sorted(set(collect_ignore))
