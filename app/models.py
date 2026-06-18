from pydantic import BaseModel
from typing import List


class AskRequest(BaseModel):
    query: str
    k: int = 4                    


class Source(BaseModel):
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    no_answer: bool
    latency_ms: float


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int