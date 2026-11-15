"""Shared prompt and output cleanup for every SQL backend."""
import re

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language "
    "question, generate a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)

# The exact template the adapter was fine tuned on. Every backend that runs
# the fine tuned model must send this raw text, not a chat template.
PROMPT_TEMPLATE = (
    "### System:\n{system}\n\n"
    "### Schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### SQL:\n"
)

STOP_SEQUENCES = ["\n\n", "###"]


def build_prompt(question: str, schema: str) -> str:
    return PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, schema=schema, question=question)


def clean_sql(raw_output: str) -> str:
    """
    Keeps the first SQL statement from a model reply and turns double quoted
    string values into single quoted ones, which is what PostgreSQL expects.
    """
    if not raw_output or not raw_output.strip():
        return ""
    text = raw_output.replace("```sql", "").replace("```", "")
    sql_lines = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            if sql_lines:
                break
            continue
        sql_lines.append(stripped)
        if stripped.endswith(";"):
            break
    sql = " ".join(sql_lines).strip()
    # Earlier versions used r"\'\1\'" here, which left literal backslashes in
    # the SQL (\'Rock\') and made PostgreSQL reject the query.
    return re.sub(r'"([^"]*)"', r"'\1'", sql)
