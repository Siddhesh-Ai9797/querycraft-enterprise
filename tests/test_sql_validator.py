import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine
from agents.sql_validator import SQLValidatorAgent
load_dotenv()

def test_valid_sql():
    engine    = create_engine(os.getenv("DATABASE_URL"))
    validator = SQLValidatorAgent(engine)
    result    = validator.validate("SELECT COUNT(*) FROM artist")
    assert result["valid"] is True
    print("✅ Valid SQL passes")

def test_rejects_drop():
    engine    = create_engine(os.getenv("DATABASE_URL"))
    validator = SQLValidatorAgent(engine)
    result    = validator.validate("DROP TABLE artist")
    assert result["valid"] is False
    print(f"✅ DROP rejected at layer: {result['layer']}")

def test_rejects_delete():
    engine    = create_engine(os.getenv("DATABASE_URL"))
    validator = SQLValidatorAgent(engine)
    result    = validator.validate("DELETE FROM artist WHERE artist_id = 1")
    assert result["valid"] is False
    print(f"✅ DELETE rejected at layer: {result['layer']}")

def test_rejects_bad_column():
    engine    = create_engine(os.getenv("DATABASE_URL"))
    validator = SQLValidatorAgent(engine)
    result    = validator.validate("SELECT nonexistent_column FROM artist")
    assert result["valid"] is False
    print(f"✅ Bad column rejected: {result['error'][:60]}")

if __name__ == "__main__":
    test_valid_sql()
    test_rejects_drop()
    test_rejects_delete()
    test_rejects_bad_column()