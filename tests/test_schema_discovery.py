import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine
from agents.schema_discovery import SchemaDiscoveryAgent
load_dotenv()

def test_schema_discovery():
    engine = create_engine(os.getenv("DATABASE_URL"))
    agent  = SchemaDiscoveryAgent(engine)
    agent.initialize()
    result = agent.discover("How many artists are there?", top_k=3)
    assert "artist" in result["tables_used"], f"Expected artist table, got {result['tables_used']}"
    assert len(result["schema"]) > 0
    print(f"✅ Schema discovery works — tables: {result['tables_used']}")

def test_schema_lists_all_tables():
    engine = create_engine(os.getenv("DATABASE_URL"))
    agent  = SchemaDiscoveryAgent(engine)
    agent.initialize()
    tables = agent.list_all_tables()
    assert len(tables) == 11, f"Expected 11 tables, got {len(tables)}"
    print(f"✅ All 11 tables cached in Chroma")

if __name__ == "__main__":
    test_schema_discovery()
    test_schema_lists_all_tables()