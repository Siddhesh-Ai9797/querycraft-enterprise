import re
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError

DANGEROUS_KEYWORDS = {
    "DROP","DELETE","INSERT","UPDATE","ALTER","CREATE",
    "TRUNCATE","GRANT","REVOKE","EXEC","EXECUTE","INTO","MERGE","REPLACE","CALL","DO"
}

class SQLValidatorAgent:
    def __init__(self, engine):
        self.engine = engine

    def validate(self, sql: str) -> dict:
        sql = sql.strip()
        if not sql:
            return self._fail("Empty SQL", "pre-check")
        for check in [self._check_select_only, self._check_dangerous_keywords, self._check_explain]:
            result = check(sql)
            if not result["valid"]:
                return result
        return self._pass(sql)

    def _check_select_only(self, sql):
        first_word = sql.split()[0].upper()
        if first_word != "SELECT":
            return self._fail(f"Only SELECT allowed. Got: {first_word}", "select-only")
        return self._pass(sql)

    def _check_dangerous_keywords(self, sql):
        sql_upper = sql.upper()
        for keyword in DANGEROUS_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return self._fail(f"Dangerous keyword: {keyword}", "keyword-blocklist")
        return self._pass(sql)

    def _check_explain(self, sql):
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text(f"EXPLAIN {sql}"))
                    conn.rollback()
            return self._pass(sql)
        except ProgrammingError as e:
            return self._fail(f"SQL validation failed: {self._clean_pg_error(str(e))}", "explain")
        except OperationalError as e:
            return self._fail(f"Database connection error during validation: {str(e)[:200]}", "explain")

    def _clean_pg_error(self, raw):
        match = re.search(r'\) (.+?)(?:\n|$)', raw)
        return match.group(1).strip() if match else raw[:200]

    def _pass(self, sql): return {"valid": True,  "sql": sql,  "error": None,  "layer": None}
    def _fail(self, err, layer): return {"valid": False, "sql": None, "error": err, "layer": layer}