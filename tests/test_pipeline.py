import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from core.pipeline import QueryCraftPipeline
load_dotenv()

def test_simple_query():
    pipeline = QueryCraftPipeline()
    pipeline.initialize()
    result = pipeline.run("How many artists are there?")
    assert result["success"] is True
    assert result["confidence"] > 0.5
    assert "275" in result["answer"]
    print(f"✅ Pipeline: {result['answer']} ({result['total_ms']}ms)")

def test_medium_query():
    pipeline = QueryCraftPipeline()
    pipeline.initialize()
    result = pipeline.run("Which artist has the most albums?")
    assert result["success"] is True
    print(f"✅ Medium query: {result['answer']}")

if __name__ == "__main__":
    test_simple_query()
    test_medium_query()