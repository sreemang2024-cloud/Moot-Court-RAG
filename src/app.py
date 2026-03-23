from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import settings
from src.llm import ABSTAIN_ANSWER, generate_grounded_answer
from src.retrieval import HybridRetriever
from src.schemas import AskRequest, AskResponse, RetrievedChunk

app = FastAPI(title="Moot Court Legal RAG", version="0.1.0")
retriever: HybridRetriever | None = None


@app.on_event("startup")
def startup_event() -> None:
    global retriever
    try:
        retriever = HybridRetriever()
    except FileNotFoundError:
        retriever = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": retriever is not None,
        "min_context_score": settings.min_context_score,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse | JSONResponse:
    if retriever is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Index not found. Run `python -m src.ingest` and `python -m src.retrieval --build` first."
            },
        )

    docs = retriever.search(payload.question, top_k=settings.top_k)
    best = max((d.score for d in docs), default=0.0)

    if best < settings.min_context_score:
        return AskResponse(
            answer=ABSTAIN_ANSWER,
            grounded=False,
            citations=[],
            contexts=[
                RetrievedChunk(
                    source=d.source,
                    chunk_id=d.chunk_id,
                    score=d.score,
                    text=d.text,
                )
                for d in docs
            ],
        )

    answer, citations = generate_grounded_answer(payload.question, docs)
    grounded = bool(citations)

    if not grounded:
        answer = ABSTAIN_ANSWER

    return AskResponse(
        answer=answer,
        grounded=grounded,
        citations=citations,
        contexts=[
            RetrievedChunk(
                source=d.source,
                chunk_id=d.chunk_id,
                score=d.score,
                text=d.text,
            )
            for d in docs
        ],
    )
