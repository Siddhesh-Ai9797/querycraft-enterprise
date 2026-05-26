import re
import mlflow

class QualityMonitor:
    WEIGHT_CONFIDENCE   = 0.50
    WEIGHT_COMPLETENESS = 0.30
    WEIGHT_COMPLEXITY   = 0.20

    def score(self, pipeline_result: dict, execution_result: dict) -> dict:
        if not pipeline_result["success"]:
            return self._zero_score("Pipeline failed")
        confidence  = pipeline_result.get("confidence", 0.5)
        sql         = pipeline_result.get("sql", "")
        row_count   = execution_result.get("row_count", 0)
        completeness_score, completeness_note = self._score_completeness(row_count, sql)
        complexity_level, complexity_score    = self._score_complexity(sql)
        quality_score = round(min(1.0, max(0.0,
            self.WEIGHT_CONFIDENCE * confidence +
            self.WEIGHT_COMPLETENESS * completeness_score +
            self.WEIGHT_COMPLEXITY * complexity_score
        )), 4)
        flagged     = quality_score < 0.6
        flag_reason = None
        if flagged:
            reasons = []
            if confidence < 0.6: reasons.append(f"low confidence ({confidence:.2f})")
            if completeness_score < 0.5: reasons.append(completeness_note)
            flag_reason = "; ".join(reasons) if reasons else "low composite score"
        return {
            "quality_score": quality_score, "confidence_score": confidence,
            "completeness_score": completeness_score, "complexity_score": complexity_score,
            "sql_complexity": complexity_level, "row_count": row_count,
            "flagged": flagged, "flag_reason": flag_reason,
        }

    def log_to_mlflow(self, pipeline_result: dict, quality_result: dict):
        mlflow.log_metric("quality_score",      quality_result["quality_score"])
        mlflow.log_metric("completeness_score", quality_result["completeness_score"])
        mlflow.log_metric("complexity_score",   quality_result["complexity_score"])
        mlflow.log_param("sql_complexity",      quality_result["sql_complexity"])
        mlflow.log_metric("flagged", 1.0 if quality_result["flagged"] else 0.0)

    def _score_completeness(self, row_count, sql):
        if any(fn in sql.upper() for fn in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
            return 1.0, "aggregation"
        if row_count == 0: return 0.2, "empty result set"
        if row_count == 1: return 0.9, "single row"
        return 1.0, "multi-row"

    def _score_complexity(self, sql):
        sql_upper    = sql.upper()
        join_count   = len(re.findall(r'\bJOIN\b', sql_upper))
        subq_count   = len(re.findall(r'\bSELECT\b', sql_upper)) - 1
        if join_count == 0 and subq_count == 0: return "simple", 1.0
        if join_count == 1 and subq_count == 0: return "medium", 0.8
        if join_count == 2 or (join_count == 1 and subq_count > 0): return "complex", 0.6
        return "complex", 0.5

    def _zero_score(self, reason):
        return {"quality_score": 0.0, "confidence_score": 0.0, "completeness_score": 0.0,
                "complexity_score": 0.0, "sql_complexity": "unknown", "row_count": 0,
                "flagged": True, "flag_reason": reason}