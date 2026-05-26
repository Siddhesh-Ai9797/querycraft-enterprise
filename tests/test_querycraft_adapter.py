import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.production_adapter import ProductionQueryAdapter
from dotenv import load_dotenv
load_dotenv()

SCHEMA = """CREATE TABLE artist (artist_id INTEGER NOT NULL, name VARCHAR(120))
CREATE TABLE album (album_id INTEGER NOT NULL, title VARCHAR(160), artist_id INTEGER NOT NULL)"""

def test_production_adapter():
    adapter = ProductionQueryAdapter()
    result  = adapter.generate("How many artists are there?", SCHEMA)
    assert result["sql"], "SQL should not be empty"
    assert "SELECT" in result["sql"].upper()
    assert result["latency_ms"] > 0
    print(f"✅ Production adapter SQL: {result['sql']}")
    print(f"   Latency: {result['latency_ms']}ms")

def test_sql_cleaning():
    adapter = ProductionQueryAdapter()
    result  = adapter.generate('List artists named "AC/DC"', SCHEMA)
    assert '"' not in result["sql"] or result["sql"].count('"') == 0
    print(f"✅ Quote cleaning: {result['sql']}")

if __name__ == "__main__":
    test_production_adapter()
    test_sql_cleaning()