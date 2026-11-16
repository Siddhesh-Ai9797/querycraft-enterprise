import os
# This project never uses TensorFlow or JAX. Telling transformers so avoids
# loading a broken TensorFlow install that may sit in the same environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from sqlalchemy import create_engine
import mlflow
load_dotenv()

EVAL_QUESTIONS = [
    "How many artists are there?",
    "What is the total revenue from all invoices?",
    "Which artist has the most albums?",
    "How many tracks are in the Rock genre?",
    "How many customers are from the USA?",
    "What is the average invoice total?",
    "How many invoices are there in total?",
    "How many albums are in the database?",
    "Which country has the most customers?",
    "What is the most expensive track price?",
]

def run_ragas_evaluation():
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness, context_precision
        from datasets import Dataset
    except ImportError:
        print("Install ragas: pip install ragas==0.1.21 datasets")
        return

    from core.pipeline import QueryCraftPipeline
    pipeline = QueryCraftPipeline()
    pipeline.initialize()

    data = {"question": [], "answer": [], "contexts": [], "ground_truths": []}
    print("Running pipeline on evaluation questions...")
    for q in EVAL_QUESTIONS:
        result = pipeline.run(q)
        data["question"].append(q)
        data["answer"].append(result.get("answer", ""))
        data["contexts"].append([result.get("schema_context", result.get("sql", ""))])
        data["ground_truths"].append([""])
        print(f"  ✅ {q[:50]}")

    dataset = Dataset.from_dict(data)
    print("\nRunning RAGAS evaluation...")
    scores = evaluate(dataset, metrics=[answer_relevancy, faithfulness, context_precision])

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("querycraft-enterprise-ragas")
    with mlflow.start_run(run_name="ragas_querycraft"):
        mlflow.log_param("model", "querycraft-enterprise")
        mlflow.log_param("sql_backend", os.getenv("INFERENCE_BACKEND", "local"))
        mlflow.log_param("n_questions", len(EVAL_QUESTIONS))
        mlflow.log_metric("answer_relevancy", scores["answer_relevancy"])
        mlflow.log_metric("faithfulness", scores["faithfulness"])
        mlflow.log_metric("context_precision", scores["context_precision"])

    print(f"\n{'='*50}")
    print(f"RAGAS Results — QueryCraft Enterprise")
    print(f"  Answer Relevancy : {scores['answer_relevancy']:.3f}")
    print(f"  Faithfulness     : {scores['faithfulness']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")
    print(f"{'='*50}")
    backend = os.getenv("INFERENCE_BACKEND", "local")
    os.makedirs("benchmarks/results", exist_ok=True)
    with open(f"benchmarks/results/ragas_{backend}.json", "w") as f:
        json.dump({"backend": backend, "n_questions": len(EVAL_QUESTIONS),
                   "answer_relevancy": float(scores["answer_relevancy"]),
                   "faithfulness": float(scores["faithfulness"]),
                   "context_precision": float(scores["context_precision"])}, f, indent=2)
    return scores

if __name__ == "__main__":
    run_ragas_evaluation()