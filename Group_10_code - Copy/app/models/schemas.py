"""
upsc-sarathi-agentic :: app/models/schemas.py
Pydantic request/response schemas for the FastAPI layer.
(Agent-internal structured outputs live in app/models/agent_schemas.py)
"""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------- request models ----------
class QARequest(BaseModel):
    question: str
    k: int = 6
    subject: Optional[str] = None
    include_pyq: bool = True


class MCQRequest(BaseModel):
    topic: str
    n: int = Field(default=10, ge=1, le=50)
    subject: Optional[str] = None


class MainsRequest(BaseModel):
    topic: str
    n: int = Field(default=5, ge=1, le=15)
    gs_paper: Optional[Literal["GS-I", "GS-II", "GS-III", "GS-IV", "Essay"]] = None


class NotesRequest(BaseModel):
    topic: str
    subject: Optional[str] = None


class PrelimsPaperRequest(BaseModel):
    topics: List[str]
    n: int = Field(default=20, ge=5, le=100)


class MainsPaperRequest(BaseModel):
    topic: str
    gs_paper: Literal["GS-I", "GS-II", "GS-III", "GS-IV"] = "GS-II"
    n: int = Field(default=5, ge=1, le=15)


class MainsWriteRequest(BaseModel):
    """Ask the agent crew to draft a model UPSC-standard mains answer."""
    question: str
    marks: int = Field(default=15, ge=10, le=20)
    gs_paper: Optional[str] = None


class EssayWriteRequest(BaseModel):
    topic: str


class AnswerEvalRequest(BaseModel):
    question: str
    answer: str


class EssayEvalRequest(BaseModel):
    topic: str
    essay_text: str


class PYQMatchRequest(BaseModel):
    topic: str
    k: int = Field(default=10, ge=3, le=30)


class SyllabusMapRequest(BaseModel):
    text: str


class SRAddRequest(BaseModel):
    """Add a new spaced-repetition card. Previously /sr/add took bare
    scalar function params, which FastAPI reads as query parameters - but
    every caller (the Streamlit frontend, this project's own conventions)
    sends a JSON body like every other endpoint. That mismatch made the
    endpoint silently fail with a 422 on every real call; this model fixes
    it by matching the rest of the API's request-model pattern."""
    topic: str
    front: str
    back: str


class SRReviewRequest(BaseModel):
    """Record a spaced-repetition review. Same fix as SRAddRequest above -
    /sr/review had the identical bare-scalar-params bug."""
    card_id: str
    quality: int = Field(..., ge=0, le=5)


# ---------- response models ----------
class Source(BaseModel):
    source: str
    page: int
    score: float
    snippet: str


class QAResponse(BaseModel):
    answer: str
    sources: List[Source]
    used_rag: bool
    route_reason: str
    agent_trace: List[dict]
