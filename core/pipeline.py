import torch
import os
import time
import mlflow
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_core.runnables import RunnableLambda
from core.quality_monitor import QualityMonitor
load_dotenv()

MAX_RETRIES = 2

class QueryCraftPipeline:
    def __init__(self):
        engine = create_engine(os.getenv("DATABASE_URL"))
        from agents.schema_discovery  import SchemaDiscoveryAgent
        from agents.sql_validator     import SQLValidatorAgent
        from agents.sql_executor      import SQLExecutorAgent
        from agents.answer_synthesizer import AnswerSynthesizerAgent
        self.schema_agent    = SchemaDiscoveryAgent(engine)
        self.validator       = SQLValidatorAgent(engine)
        self.executor        = SQLExecutorAgent(engine)
        self.synthesizer     = AnswerSynthesizerAgent()
        self.quality_monitor = QualityMonitor()
        inference_backend = os.getenv("INFERENCE_BACKEND", "local")
        if inference_backend == "openai":
            from core.production_adapter import ProductionQueryAdapter
            self.adapter = ProductionQueryAdapter()
            print("  Using production adapter (GPT-3.5-turbo)")
        else:
            from core.querycraft_adapter import QueryCraftAdapter
            self.adapter = QueryCraftAdapter()
            print("  Using local adapter (Phi-3 Mini)")
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("querycraft-enterprise")
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        print("Initializing QueryCraft Enterprise pipeline...")
        self.schema_agent.initialize()
        self.adapter.load()
        self._schema_step     = RunnableLambda(self._schema_step_fn)
        self._generate_step   = RunnableLambda(self._generate_step_fn)
        self._validate_step   = RunnableLambda(self._validate_step_fn)
        self._execute_step    = RunnableLambda(self._execute_step_fn)
        self._synthesize_step = RunnableLambda(self._synthesize_step_fn)
        self._initialized = True
        print("✅ Pipeline ready\n")

    def run(self, question: str) -> dict:
        if not self._initialized:
            self.initialize()
        pipeline_start = time.time()
        with mlflow.start_run(run_name=f"query_{int(time.time())}"):
            mlflow.log_param("question", question[:250])
            schema_result  = self._schema_step.invoke({"question": question})
            mlflow.log_param("tables_used", str(schema_result["tables_used"]))
            schema         = schema_result["schema"]
            sql            = None
            attempts       = 0
            error_feedback = None
            while attempts < MAX_RETRIES:
                attempts += 1
                gen_result    = self._generate_step.invoke(
                    {"question": question, "schema": schema, "error_feedback": error_feedback})
                candidate_sql = gen_result["sql"]
                mlflow.log_metric(f"attempt_{attempts}_latency_ms", gen_result["latency_ms"])
                mlflow.log_param(f"attempt_{attempts}_sql", candidate_sql[:250])
                val_result = self._validate_step.invoke({"sql": candidate_sql})
                if val_result["valid"]:
                    sql = candidate_sql
                    break
                else:
                    error_feedback = val_result["error"]
                    mlflow.log_param(f"attempt_{attempts}_error", error_feedback[:250])
                    if attempts < MAX_RETRIES:
                        print(f"  ⚠️  Attempt {attempts} failed: {error_feedback}")
                        print(f"  🔄 Retrying with error feedback...")
            if sql is None:
                total_ms = int((time.time()-pipeline_start)*1000)
                mlflow.log_metric("success", 0)
                return self._error_result(question, error_feedback, attempts, total_ms)
            exec_result  = self._execute_step.invoke({"sql": sql})
            mlflow.log_metric("row_count",    exec_result["row_count"])
            mlflow.log_metric("execution_ms", exec_result["execution_ms"])
            synth_result = self._synthesize_step.invoke(
                {"question": question, "sql": sql, "execution_result": exec_result})
            total_ms     = int((time.time()-pipeline_start)*1000)
            quality_result = self.quality_monitor.score(
                {"success": True, "confidence": synth_result["confidence"], "sql": sql},
                exec_result)
            mlflow.log_metric("confidence",   synth_result["confidence"])
            mlflow.log_metric("total_ms",     total_ms)
            mlflow.log_metric("tokens_used",  synth_result["tokens_used"])
            mlflow.log_metric("attempts",     attempts)
            mlflow.log_metric("success",      1)
            self.quality_monitor.log_to_mlflow({}, quality_result)
            return {
                "question": question, "sql": sql,
                "answer": synth_result["answer"], "confidence": synth_result["confidence"],
                "quality_score": quality_result["quality_score"],
                "sql_complexity": quality_result["sql_complexity"],
                "flagged": quality_result["flagged"],
                "tables_used": schema_result["tables_used"],
                "schema_context": schema_result["schema"],
                "total_ms": total_ms, "attempts": attempts, "success": True, "error": None,
            }

    def _schema_step_fn(self, inputs):   return self.schema_agent.discover(inputs["question"])
    def _generate_step_fn(self, inputs):
        q = inputs["question"]
        if inputs.get("error_feedback"):
            q += f"\n\n[Previous error: {inputs['error_feedback']}. Please correct.]"
        return self.adapter.generate(q, inputs["schema"])
    def _validate_step_fn(self, inputs): return self.validator.validate(inputs["sql"])
    def _execute_step_fn(self, inputs):  return self.executor.execute(inputs["sql"])
    def _synthesize_step_fn(self, inputs):
        return self.synthesizer.synthesize(
            inputs["question"], inputs["sql"], inputs["execution_result"])
    def _error_result(self, question, error, attempts, total_ms):
        return {"question": question, "sql": None,
                "answer": f"Unable to generate valid query. Error: {error}",
                "confidence": 0.0, "quality_score": 0.0, "sql_complexity": "unknown",
                "flagged": True, "tables_used": [], "total_ms": total_ms,
                "attempts": attempts, "success": False, "error": error}