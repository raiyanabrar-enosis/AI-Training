from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Document RAG API",
    description="Semantic search and grounded question answering over PDFs",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str
    limit: int = 5
    source: str | None = None


class Source(BaseModel):
    text: str
    page: int
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}