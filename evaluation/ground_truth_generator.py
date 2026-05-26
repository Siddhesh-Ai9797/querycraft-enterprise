import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from decimal import Decimal
load_dotenv()

EVAL_SET = [
    ("How many artists are there?", "SELECT COUNT(*) FROM artist", "simple"),
    ("How many tracks are there in total?", "SELECT COUNT(*) FROM track", "simple"),
    ("What is the total revenue from all invoices?", "SELECT SUM(total) FROM invoice", "simple"),
    ("How many customers are there?", "SELECT COUNT(*) FROM customer", "simple"),
    ("How many genres exist in the database?", "SELECT COUNT(*) FROM genre", "simple"),
    ("What is the average invoice total?", "SELECT ROUND(AVG(total)::numeric, 2) FROM invoice", "simple"),
    ("How many invoices are there in total?", "SELECT COUNT(*) FROM invoice", "simple"),
    ("How many customers are from the USA?", "SELECT COUNT(*) FROM customer WHERE country = 'USA'", "simple"),
    ("What is the most expensive track price?", "SELECT MAX(unit_price) FROM track", "simple"),
    ("How many albums are in the database?", "SELECT COUNT(*) FROM album", "simple"),
    ("How many albums does Iron Maiden have?",
     "SELECT COUNT(*) FROM album a JOIN artist ar ON a.artist_id = ar.artist_id WHERE ar.name = 'Iron Maiden'", "medium"),
    ("Which artist has the most albums?",
     "SELECT ar.name FROM artist ar JOIN album a ON ar.artist_id = a.artist_id GROUP BY ar.name ORDER BY COUNT(*) DESC LIMIT 1", "medium"),
    ("How many tracks are in the Rock genre?",
     "SELECT COUNT(*) FROM track t JOIN genre g ON t.genre_id = g.genre_id WHERE g.name = 'Rock'", "medium"),
    ("What is the most popular genre by number of tracks?",
     "SELECT g.name FROM genre g JOIN track t ON g.genre_id = t.genre_id GROUP BY g.name ORDER BY COUNT(*) DESC LIMIT 1", "medium"),
    ("Which country has the most customers?",
     "SELECT country FROM customer GROUP BY country ORDER BY COUNT(*) DESC LIMIT 1", "medium"),
    ("How many tracks does Led Zeppelin have?",
     "SELECT COUNT(*) FROM track t JOIN album a ON t.album_id = a.album_id JOIN artist ar ON a.artist_id = ar.artist_id WHERE ar.name = 'Led Zeppelin'", "medium"),
    ("Which employee supports the most customers?",
     "SELECT e.first_name, e.last_name FROM employee e JOIN customer c ON e.employee_id = c.support_rep_id GROUP BY e.employee_id ORDER BY COUNT(*) DESC LIMIT 1", "medium"),
    ("Which album has the most tracks?",
     "SELECT a.title FROM album a JOIN track t ON a.album_id = t.album_id GROUP BY a.album_id, a.title ORDER BY COUNT(*) DESC LIMIT 1", "medium"),
    ("Which customer has spent the most money?",
     "SELECT c.first_name, c.last_name, ROUND(SUM(i.total)::numeric, 2) FROM customer c JOIN invoice i ON c.customer_id = i.customer_id GROUP BY c.customer_id ORDER BY SUM(i.total) DESC LIMIT 1", "hard"),
    ("How many tracks are longer than 5 minutes?",
     "SELECT COUNT(*) FROM track WHERE milliseconds > 300000", "hard"),
]

def generate_ground_truth(engine):
    results = []
    with engine.connect() as conn:
        for question, sql, difficulty in EVAL_SET:
            try:
                result = conn.execute(text(sql))
                columns = list(result.keys())
                rows = result.fetchall()
                if len(rows) == 1 and len(columns) == 1:
                    val = rows[0][0]
                    if isinstance(val, Decimal): val = float(val)
                    ground_truth = str(val)
                    context_text = f"Query result: {val}"
                else:
                    formatted = []
                    for row in rows[:5]:
                        rd = {}
                        for col, val in zip(columns, row):
                            if isinstance(val, Decimal): val = float(val)
                            rd[col] = val
                        formatted.append(str(rd))
                    ground_truth = "; ".join(formatted[:2])
                    context_text = "\n".join(formatted)
                results.append({"question": question, "ground_truth": ground_truth,
                                 "reference_sql": sql, "context_text": context_text,
                                 "difficulty": difficulty, "success": True})
            except Exception as e:
                results.append({"question": question, "ground_truth": "", "reference_sql": sql,
                                 "context_text": "", "difficulty": difficulty,
                                 "success": False, "error": str(e)})
    return results

if __name__ == "__main__":
    engine = create_engine(os.getenv("DATABASE_URL"))
    results = generate_ground_truth(engine)
    passed = sum(1 for r in results if r["success"])
    print(f"Generated ground truth for {passed}/{len(results)} questions")
    for r in results[:5]:
        print(f"  [{r['difficulty']}] {r['question']}")
        print(f"  → {r['ground_truth']}")