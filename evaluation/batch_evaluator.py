import os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import QueryCraftPipeline

EVAL_QUESTIONS = [
    ("How many artists are there?", "simple", "275"),
    ("How many tracks are there in total?", "simple", "3503"),
    ("What is the total revenue from all invoices?", "simple", "2328"),
    ("How many customers are there?", "simple", "59"),
    ("How many genres exist in the database?", "simple", "25"),
    ("How many albums does Iron Maiden have?", "medium", "21"),
    ("How many tracks are in the Rock genre?", "medium", None),
    ("What is the average invoice total?", "medium", None),
    ("How many customers are from the USA?", "medium", None),
    ("Which artist has the most albums?", "medium", "Iron Maiden"),
    ("What is the most popular genre by number of tracks?", "medium", None),
    ("How many invoices are there in total?", "medium", None),
    ("Which country has the most customers?", "medium", None),
    ("Which customer has spent the most money?", "hard", None),
    ("Which artist has generated the most revenue?", "hard", None),
    ("What are the top 3 selling genres by revenue?", "hard", None),
    ("Which employee supports the most customers?", "hard", None),
    ("What is the most expensive track?", "hard", None),
    ("Which album has the most tracks?", "hard", None),
    ("What is the total revenue per country?", "hard", None),
]

def run_batch_evaluation():
    print("=" * 60)
    print("QueryCraft Enterprise — Batch Evaluation (20 questions)")
    print("=" * 60)
    pipeline = QueryCraftPipeline()
    pipeline.initialize()
    results, passed, failed = [], 0, 0
    for i, (question, difficulty, expected_keyword) in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i:02d}/20] [{difficulty.upper()}] {question}")
        try:
            result = pipeline.run(question)
            keyword_match = True
            if expected_keyword and result["answer"]:
                keyword_match = expected_keyword.lower() in result["answer"].lower()
            status = "✅" if result["success"] and keyword_match else "⚠️"
            if result["success"]: passed += 1
            else: failed += 1
            sql_display = result["sql"][:70] + "..." if result["sql"] and len(result["sql"]) > 70 else result["sql"]
            ans_display = result.get("answer","")[:80]
            print(f"  SQL    : {sql_display}")
            print(f"  Answer : {ans_display}")
            print(f"  Quality: {result.get('quality_score','N/A')} | {result['total_ms']}ms {status}")
            results.append({**result, "difficulty": difficulty, "keyword_match": keyword_match})
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR: {str(e)[:100]}")
            results.append({"question": question, "difficulty": difficulty, "success": False, "error": str(e)[:200]})
        time.sleep(0.5)
    print("\n" + "=" * 60)
    print(f"  Passed: {passed}  Failed: {failed}  Success rate: {passed/len(EVAL_QUESTIONS)*100:.1f}%")
    with open("evaluation/batch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_batch_evaluation()