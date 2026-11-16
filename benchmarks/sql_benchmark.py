import os
# This project never uses TensorFlow or JAX. Telling transformers so avoids
# loading a broken TensorFlow install that may sit in the same environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
"""
Fresh SQL benchmark for every way QueryCraft can run.

Uses held out examples from b-mc2/sql-create-context, split exactly like
training (shuffle with seed 42, first 2,000 rows held out), so no backend has
seen these questions.

For each backend it measures:
    exact match         SQL text equals the gold SQL after lowercasing and
                        collapsing spaces, the same rule as the original eval
    execution accuracy  the SQL runs on the given schema in SQLite
    latency             one request at a time, p50 and p95
    throughput          requests per second with several requests at once
    cost per 1,000      from token prices for OpenAI, or from the hourly price
                        of the hardware divided by measured throughput

Examples:
    python benchmarks/sql_benchmark.py --backends local,vllm,llamacpp,openai -n 500
    python benchmarks/sql_benchmark.py --backends modal -n 500
Results go to benchmarks/results/.
"""
# torch must load before datasets and onnxruntime on Windows, or its DLLs fail
# with WinError 1114. Same rule as core/pipeline.py.
try:
    import torch  # noqa: F401
except ImportError:
    pass
import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Dollars per hour for the hardware behind each self hosted backend.
# Check the provider's pricing page if these change.
HOURLY_PRICE = {
    "modal"   : float(os.getenv("MODAL_L4_PER_HOUR", "0.80")),     # Modal L4 GPU
    "llamacpp": float(os.getenv("EC2_T3_LARGE_PER_HOUR", "0.0832")),  # AWS t3.large, 2 vCPU
}
# Dollars per million tokens, input and output.
TOKEN_PRICE = {
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-4o-mini"  : (0.15, 0.60),
    "gpt-4o"       : (2.50, 10.00),
}


# ── Data and scoring ──────────────────────────────────────────────────────
def load_held_out(n: int) -> list[dict]:
    from datasets import load_dataset
    raw = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=42)
    held_out = raw.select(range(2000))
    return [{"question": r["question"], "context": r["context"], "answer": r["answer"]}
            for r in held_out.select(range(min(n, 2000)))]


def normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").lower().strip()).rstrip(";").strip()


def full_statement(raw: str) -> str:
    """
    The whole first SQL statement in a reply, joined onto one line. The fine
    tuned model answers on one line; chat models often spread a query over
    several lines, and scoring only the first line would cut them off.
    """
    text = (raw or "").replace("```sql", "").replace("```", "").strip()
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if lines:
                break
            continue
        lines.append(line)
        if line.endswith(";"):
            break
    return " ".join(lines).rstrip(";").strip()


def executes(sql: str, schema: str) -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        for stmt in (s.strip() for s in schema.split(";")):
            if stmt.upper().startswith("CREATE"):
                conn.execute(stmt)
        conn.execute(sql).fetchall()
        conn.close()
        return True
    except (sqlite3.Error, ValueError):
        return False


def pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


# ── Backends ──────────────────────────────────────────────────────────────
def make_adapter(name: str):
    if name == "local":
        from core.querycraft_adapter import QueryCraftAdapter
        return QueryCraftAdapter()
    if name in ("vllm", "modal", "llamacpp"):
        from core.remote_sql_adapter import RemoteSQLAdapter
        return RemoteSQLAdapter(name)   # reads VLLM_URL, MODAL_URL or LLAMACPP_URL
    if name in TOKEN_PRICE:
        os.environ["OPENAI_SQL_MODEL"] = name
        import importlib
        import core.production_adapter as pa
        importlib.reload(pa)
        return pa.ProductionQueryAdapter()
    raise ValueError(f"Unknown backend {name}")


def run_backend(name: str, examples: list[dict], concurrency: int, throughput_n: int) -> dict:
    print(f"\n=== {name} ===")
    adapter = make_adapter(name)
    adapter.load()
    adapter.generate(examples[0]["question"], examples[0]["context"])   # warm up

    rows, latencies, tok_in, tok_out = [], [], 0, 0
    for i, ex in enumerate(examples, 1):
        out = adapter.generate(ex["question"], ex["context"])
        sql = full_statement(out.get("raw_output", ""))
        em = normalize(sql) == normalize(ex["answer"])
        ok = executes(sql, ex["context"])
        latencies.append(out["latency_ms"])
        tok_in += out.get("prompt_tokens") or 0
        tok_out += out.get("completion_tokens") or 0
        rows.append({"question": ex["question"], "gold": ex["answer"], "predicted": sql,
                     "exact_match": em, "executes": ok, "latency_ms": out["latency_ms"]})
        if i % 50 == 0:
            done = sum(r["exact_match"] for r in rows)
            print(f"  {i}/{len(examples)}  exact match so far {100 * done / i:.1f}%")

    n = len(rows)
    result = {
        "backend"           : name,
        "n"                 : n,
        "exact_match_pct"   : round(100 * sum(r["exact_match"] for r in rows) / n, 1),
        "execution_pct"     : round(100 * sum(r["executes"] for r in rows) / n, 1),
        "latency_p50_ms"    : pct(latencies, 50),
        "latency_p95_ms"    : pct(latencies, 95),
        "avg_prompt_tokens" : round(tok_in / n, 1) if tok_in else None,
    }

    # Throughput with several requests at once, only for servers that can take them.
    if name in ("vllm", "modal", "llamacpp") or name in TOKEN_PRICE:
        batch = (examples * ((throughput_n // len(examples)) + 1))[:throughput_n]
        start = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda ex: adapter.generate(ex["question"], ex["context"]), batch))
        elapsed = time.time() - start
        result["throughput_rps"] = round(throughput_n / elapsed, 2)
        result["concurrency"] = concurrency

    if name in TOKEN_PRICE:
        price_in, price_out = TOKEN_PRICE[name]
        per_query = (tok_in * price_in + tok_out * price_out) / 1_000_000 / n
        result["cost_per_1k_usd"] = round(per_query * 1000, 4)
        result["cost_basis"] = "OpenAI token prices"
    elif name in HOURLY_PRICE and result.get("throughput_rps"):
        per_query = HOURLY_PRICE[name] / 3600 / result["throughput_rps"]
        result["cost_per_1k_usd"] = round(per_query * 1000, 4)
        result["cost_basis"] = f"${HOURLY_PRICE[name]}/hour at measured throughput, GPU busy"
    else:
        result["cost_per_1k_usd"] = None
        result["cost_basis"] = "own hardware"

    print(json.dumps(result, indent=2))
    return {"summary": result, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="local,vllm,llamacpp,gpt-4o-mini",
                        help="Comma list from: local, vllm, llamacpp, modal, gpt-4o-mini, gpt-4o")
    parser.add_argument("-n", type=int, default=500, help="Held out examples, up to 2000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--throughput-n", type=int, default=100)
    args = parser.parse_args()

    examples = load_held_out(args.n)
    os.makedirs("benchmarks/results", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    summaries = []
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        try:
            out = run_backend(name, examples, args.concurrency, args.throughput_n)
        except Exception as e:  # keep going so one broken backend doesn't lose the rest
            print(f"  ❌ {name} failed: {e}")
            summaries.append({"backend": name, "error": str(e)[:300]})
            continue
        summaries.append(out["summary"])
        with open(f"benchmarks/results/sql_{name}_{stamp}.json", "w") as f:
            json.dump(out, f, indent=2)

    with open(f"benchmarks/results/sql_summary_{stamp}.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\n| Backend | Exact match | Runs | p50 | p95 | Req/s | Cost per 1K |")
    print("|---|---|---|---|---|---|---|")
    for s in summaries:
        if "error" in s:
            print(f"| {s['backend']} | failed: {s['error'][:60]} |||||| ")
            continue
        cost = f"${s['cost_per_1k_usd']}" if s.get("cost_per_1k_usd") is not None else "own GPU"
        print(f"| {s['backend']} | {s['exact_match_pct']}% | {s['execution_pct']}% | "
              f"{s['latency_p50_ms']}ms | {s['latency_p95_ms']}ms | {s.get('throughput_rps', '–')} | {cost} |")


if __name__ == "__main__":
    main()
