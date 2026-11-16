# Serving image for the hosted demo. The fine tuned model runs on Modal (or
# vLLM, llama.cpp, OpenAI), so this container only runs the agents and the
# web page. No torch, no GPU.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
# Bake in the table search model and the Chinook data so startup needs no downloads.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])"
RUN mkdir -p data && curl -fsSL -o data/chinook.sql \
    https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql.sql
COPY . .
ENV INFERENCE_BACKEND=modal \
    SEED_CHINOOK=1 \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 CMD curl -f http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
