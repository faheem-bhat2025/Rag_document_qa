import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from rag import Rag
from database import Database
from models import AskRequest, AskResponse, IngestResponse
from config import SCORE_THRESHOLD, NO_ANSWER_PHRASE, DB_PATH

rag = Rag()
database = Database(DB_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: create DB table, and load an existing index if present."""
    try:
        rag.load()
        print("Loaded existing index from disk.")
    except Exception:
        print("No index on disk yet — call /ingest first.")
    yield


app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Parse, chunk, embed the PDF, build the FAISS index, and persist it."""
    n = rag.ingest()
    rag.save()
    return IngestResponse(status="ingested", chunks_indexed=n)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Run the RAG pipeline for one query, log it, return answer + sources."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not rag.is_ready():
        raise HTTPException(status_code=409, detail="No document ingested yet. Call /ingest first.")

    start = time.time()
    retrieved = rag.retrieve(req.query, k=req.k)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    # Layer 1: score threshold short-circuit clearly out-of-scope queries
    if top_score < SCORE_THRESHOLD:
        answer, no_answer, sources = NO_ANSWER_PHRASE, True, []
    else:
        answer = rag.generate_answer(req.query, retrieved)
        # Layer 2: prompt fallback model said it couldn't find it
        no_answer = NO_ANSWER_PHRASE.lower() in answer.lower()
        sources = [] if no_answer else retrieved

    latency_ms = (time.time() - start) * 1000
    database.log_query(req.query, answer, no_answer, top_score, latency_ms)

    return AskResponse(answer=answer, sources=sources,
                       no_answer=no_answer, latency_ms=round(latency_ms, 2))


@app.get("/analytics")
def analytics():
    """Return usage analytics computed from the SQL logs."""
    return database.get_analytics()