FROM python:3.12-slim

WORKDIR /app

# Install dependencies first — this layer caches unless requirements change,
# so code edits don't trigger a full reinstall on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vector_databases/ ./vector_databases/
COPY rag/ ./rag/
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]