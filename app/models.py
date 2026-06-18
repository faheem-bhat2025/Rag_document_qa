from pydantic import BaseModel
from typing import List


class AskRequest(BaseModel):
    query: str
    k: int = 4  # number of chunks to retrieve from FAISS; caller can override up to 10


class Source(BaseModel):
    text: str
    score: float  # cosine similarity score (0.0 to 1.0); higher means closer match


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    no_answer: bool  # True when retrieval score is too low OR the LLM returned the fallback phrase
    latency_ms: float


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int