import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

st.set_page_config(page_title="QueryCraft Enterprise", page_icon="🔍", layout="wide")
MLFLOW_URI = "./mlruns"
EXPERIMENT = "querycraft-enterprise"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

@st.cache_data(ttl=30)
def load_runs():
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if not experiment: return pd.DataFrame()
        runs = client.search_runs([experiment.experiment_id],
                                   order_by=["attributes.start_time DESC"], max_results=200)
        rows = []
        for run in runs:
            m, p = run.data.metrics, run.data.params
            rows.append({"run_id": run.info.run_id[:8],
                "timestamp": datetime.fromtimestamp(run.info.start_time/1000).strftime("%Y-%m-%d %H:%M:%S"),
                "question": p.get("question","")[:60], "sql_complexity": p.get("sql_complexity","unknown"),
                "success": m.get("success",0), "confidence": m.get("confidence",None),
                "quality_score": m.get("quality_score",None), "total_ms": m.get("total_ms",None),
                "tokens_used": m.get("tokens_used",None), "flagged": m.get("flagged",0), "attempts": m.get("attempts",1)})
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"MLflow error: {e}"); return pd.DataFrame()

st.title("🔍 QueryCraft Enterprise")
st.caption("Live monitoring dashboard — auto-refreshes every 30s")
st.divider()
df = load_runs()
if df.empty:
    st.warning("No runs found."); st.stop()
df_m = df[df["total_ms"].notna()].copy()
if df_m.empty:
    st.warning("No completed runs."); st.stop()
c1,c2,c3,c4,c5 = st.columns(5)
total = len(df_m); success_count = int(df_m["success"].sum())
c1.metric("Total Queries", total)
c2.metric("Success Rate", f"{success_count/total*100:.1f}%")
c3.metric("Flagged", int(df_m["flagged"].sum()))
avg_q = df_m["quality_score"].dropna().mean()
c4.metric("Avg Quality", f"{avg_q:.3f}" if pd.notna(avg_q) else "N/A")
avg_l = df_m["total_ms"].dropna().mean()
c5.metric("Avg Latency", f"{avg_l:.0f}ms" if pd.notna(avg_l) else "N/A")
st.divider()
cl, cr = st.columns(2)
with cl:
    st.subheader("Quality Score Over Time")
    qdf = df_m[df_m["quality_score"].notna()].sort_values("timestamp")
    if not qdf.empty:
        st.line_chart(qdf[["timestamp","quality_score"]].set_index("timestamp"))
        if len(qdf)>=5:
            r_avg = qdf["quality_score"].tail(5).mean(); o_avg = qdf["quality_score"].mean()
            if o_avg-r_avg>0.05: st.warning(f"⚠️ Quality drift detected (recent: {r_avg:.3f})")
            else: st.success(f"✅ No quality drift (recent avg: {r_avg:.3f})")
with cr:
    st.subheader("Latency Distribution")
    ldf = df_m["total_ms"].dropna()
    if not ldf.empty:
        lc1,lc2,lc3 = st.columns(3)
        lc1.metric("p50", f"{ldf.quantile(0.50):.0f}ms")
        lc2.metric("p95", f"{ldf.quantile(0.95):.0f}ms")
        lc3.metric("p99", f"{ldf.quantile(0.99):.0f}ms")
        st.bar_chart(ldf.value_counts(bins=10).sort_index())
st.divider()
ca, cb = st.columns(2)
with ca:
    st.subheader("Queries by Complexity")
    cc = df_m["sql_complexity"].value_counts()
    if not cc.empty: st.bar_chart(cc)
with cb:
    st.subheader("Confidence Score Distribution")
    cd = df_m["confidence"].dropna()
    if not cd.empty: st.bar_chart(cd.value_counts(bins=10).sort_index())
st.divider()
st.subheader("Recent Query Log")
disp = df_m[["timestamp","question","sql_complexity","quality_score","confidence","total_ms","attempts","flagged"]].head(20).copy()
disp["flagged"] = disp["flagged"].apply(lambda x: "⚠️" if x==1.0 else "✅")
disp["quality_score"] = disp["quality_score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
disp["total_ms"] = disp["total_ms"].apply(lambda x: f"{x:.0f}ms" if pd.notna(x) else "N/A")
st.dataframe(disp, use_container_width=True, hide_index=True)
st.divider()
st.caption(f"MLflow: {MLFLOW_URI} | Experiment: {EXPERIMENT} | {datetime.now().strftime('%H:%M:%S')}")