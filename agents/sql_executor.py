import time
from decimal import Decimal
from datetime import datetime, date
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError

class SQLExecutorAgent:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, sql: str) -> dict:
        start = time.time()
        try:
            with self.engine.connect() as conn:
                result   = conn.execute(text(sql))
                columns  = list(result.keys())
                raw_rows = result.fetchall()
            execution_ms = int((time.time() - start) * 1000)
            rows = []
            for raw_row in raw_rows:
                row_dict = {}
                for col, val in zip(columns, raw_row):
                    row_dict[col] = self._serialize(val)
                rows.append(row_dict)
            return {"success": True, "rows": rows, "columns": columns,
                    "row_count": len(rows), "execution_ms": execution_ms, "error": None}
        except (ProgrammingError, OperationalError) as e:
            return {"success": False, "rows": [], "columns": [], "row_count": 0,
                    "execution_ms": int((time.time()-start)*1000), "error": str(e)[:300]}

    def _serialize(self, value):
        if isinstance(value, Decimal): return float(value)
        if isinstance(value, (datetime, date)): return value.isoformat()
        return value