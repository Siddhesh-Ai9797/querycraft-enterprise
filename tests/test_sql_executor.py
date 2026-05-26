import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine
from agents.sql_executor import SQLExecutorAgent
load_dotenv()

def test_count_query():
    engine   = create_engine(os.getenv("DATABASE_URL"))
    executor = SQLExecutorAgent(engine)
    result   = executor.execute("SELECT COUNT(*) AS total FROM artist")
    assert result["success"] is True
    assert result["rows"][0]["total"] == 275
    print(f"✅ Count query: {result['rows'][0]}")

def test_multirow_query():
    engine   = create_engine(os.getenv("DATABASE_URL"))
    executor = SQLExecutorAgent(engine)
    result   = executor.execute("SELECT name FROM artist ORDER BY name LIMIT 5")
    assert result["success"] is True
    assert result["row_count"] == 5
    print(f"✅ Multi-row query: {result['row_count']} rows in {result['execution_ms']}ms")

def test_decimal_serialization():
    engine   = create_engine(os.getenv("DATABASE_URL"))
    executor = SQLExecutorAgent(engine)
    result   = executor.execute("SELECT SUM(total) AS revenue FROM invoice")
    assert result["success"] is True
    revenue = result["rows"][0]["revenue"]
    assert isinstance(revenue, float), f"Expected float, got {type(revenue)}"
    print(f"✅ Decimal serialized to float: {revenue}")

if __name__ == "__main__":
    test_count_query()
    test_multirow_query()
    test_decimal_serialization()