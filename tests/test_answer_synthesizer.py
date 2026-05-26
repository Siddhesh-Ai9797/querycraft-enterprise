import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from agents.answer_synthesizer import AnswerSynthesizerAgent
load_dotenv()

def test_basic_synthesis():
    agent  = AnswerSynthesizerAgent()
    result = agent.synthesize(
        question="How many artists are there?",
        sql="SELECT COUNT(*) FROM artist",
        execution_result={"success": True, "rows": [{"count": 275}],
                          "columns": ["count"], "row_count": 1}
    )
    assert result["confidence"] > 0.5
    assert "275" in result["answer"]
    print(f"✅ Synthesis: {result['answer']}")

def test_empty_result():
    agent  = AnswerSynthesizerAgent()
    result = agent.synthesize(
        question="Who?", sql="SELECT * FROM artist WHERE 1=0",
        execution_result={"success": True, "rows": [], "columns": [], "row_count": 0}
    )
    assert result["confidence"] <= 0.5
    print(f"✅ Empty result handled: {result['answer']}")

if __name__ == "__main__":
    test_basic_synthesis()
    test_empty_result()