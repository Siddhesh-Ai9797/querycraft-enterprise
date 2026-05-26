import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine
from agents.schema_discovery import SchemaDiscoveryAgent
load_dotenv()

def test_chroma_persistence():
    engine = create_engine(os.getenv("DATABASE_URL"))
    agent1 = SchemaDiscoveryAgent(engine)
    agent1.initialize()
    count1 = agent1.collection.count()
    agent2 = SchemaDiscoveryAgent(engine)
    agent2.initialize()
    count2 = agent2.collection.count()
    assert count1 == count2 == 11
    print(f"✅ Chroma persistence: {count1} tables cached across instances")

def test_semantic_search():
    engine = create_engine(os.getenv("DATABASE_URL"))
    agent  = SchemaDiscoveryAgent(engine)
    agent.initialize()
    result = agent.discover("total sales revenue by country", top_k=3)
    tables = result["tables_used"]
    assert any(t in tables for t in ["invoice", "invoice_line", "customer"])
    print(f"✅ Semantic search for revenue: {tables}")

if __name__ == "__main__":
    test_chroma_persistence()
    test_semantic_search()