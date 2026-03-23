from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=4)


class RetrievedChunk(BaseModel):
    source: str
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: List[str]
    contexts: List[RetrievedChunk]
