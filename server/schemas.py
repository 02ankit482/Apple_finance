"""
server/schemas.py

Pydantic models for all API request and response bodies.
Strict validation keeps the API contract explicit and self-documenting.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator
import uuid


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's financial question about Apple.",
        examples=["What was Apple's revenue in FY2023?"],
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID identifying the conversation session. "
                    "Omit to start a new session.",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class DeleteSessionRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    source: str
    section: Optional[str] = None
    year: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    route: str = Field(description="Which data source was used: vectorstore | web_search | clarify")
    citations: list[str] = Field(default_factory=list)
    rewrite_count: int = Field(default=0, description="Number of query rewrites performed")
    docs_evaluated: int = Field(default=0, description="Number of chunks graded")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "question": "What was Apple's net income in FY2023?",
                "answer": "Apple reported net income of $96.995 billion in FY2023 [ITEM_7 | 2023].",
                "route": "vectorstore",
                "citations": ["[ITEM_7 | 2023 | 10k_0_item_7.txt]"],
                "rewrite_count": 0,
                "docs_evaluated": 6,
            }
        }


class StreamChunk(BaseModel):
    """Individual chunk sent over SSE."""
    type: str           # "token" | "metadata" | "done" | "error"
    content: str = ""
    metadata: Optional[dict] = None


class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    created_at: float
    last_active: float
    turns: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    vector_store_docs: int
    session_count: int
    version: str = "1.0.0"