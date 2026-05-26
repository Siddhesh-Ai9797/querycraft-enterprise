import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
load_dotenv()

def test_connection():
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM artist"))
        count = result.scalar()
    assert count == 275, f"Expected 275 artists, got {count}"
    print(f"✅ Database connected — {count} artists")

def test_all_tables():
    engine = create_engine(os.getenv("DATABASE_URL"))
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema="public")
    assert len(tables) == 11, f"Expected 11 tables, got {len(tables)}"
    print(f"✅ All 11 tables present: {sorted(tables)}")

if __name__ == "__main__":
    test_connection()
    test_all_tables()