from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json

from vector_databases.config import GENERATION_MODEL
from vector_databases.semantic_search import search_filtered, search
from rag.prompts import SYSTEM_INSTRUCTION, build_input
from rag.generator import client

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


class SearchResponse(BaseModel):
    results: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/stream")
def ask_stream_endpoint(req: AskRequest):
    if req.source:
        hits = search_filtered(req.question, source=req.source, limit=req.limit)
    else:
        hits = search(req.question, limit=req.limit)

    def event_generator():
        # Send sources first so the client can render them immediately.
        yield f"data: {json.dumps({'type': 'sources', 'sources': hits})}\n\n"

        stream = client.interactions.create(
            model=GENERATION_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            input=build_input(req.question, hits),
            stream=True,
        )

        for event in stream:
            if event.event_type == "step.delta":
                if event.delta.type == "text":
                    payload = {"type": "text", "text": event.delta.text}
                    yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
