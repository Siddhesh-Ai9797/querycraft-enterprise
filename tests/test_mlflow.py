import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlflow
from mlflow.tracking import MlflowClient

def test_mlflow_logging():
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("querycraft-enterprise-test")
    with mlflow.start_run(run_name="test_run"):
        mlflow.log_metric("test_metric", 0.99)
        mlflow.log_param("test_param", "hello")
    print("✅ MLflow logging works")

def test_mlflow_query():
    mlflow.set_tracking_uri("./mlruns")
    client     = MlflowClient()
    experiment = client.get_experiment_by_name("querycraft-enterprise")
    if experiment:
        runs = client.search_runs([experiment.experiment_id], max_results=5)
        print(f"✅ MLflow query: found {len(runs)} recent runs in querycraft-enterprise")
    else:
        print("⚠️  No querycraft-enterprise experiment yet — run the pipeline first")

if __name__ == "__main__":
    test_mlflow_logging()
    test_mlflow_query()