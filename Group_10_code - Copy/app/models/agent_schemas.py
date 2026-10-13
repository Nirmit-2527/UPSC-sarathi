"""
upsc-sarathi-agentic :: app/models/agent_schemas.py

Structured outputs produced by CrewAI agents (via Task.output_pydantic).

Why structured outputs instead of free-text markdown everywhere?
  1. The UPSC mains-answer format (intro / point-wise body+evidence /
     conclusion) is something we can *enforce programmatically* if the
     agent is forced to fill discrete fields, rather than hoping a prompt
     instruction is followed.
  2. Guardrails (app/agents/guardrails.py) need machine-readable fields
     (e.g. body_points[i].evidence) to validate "every point is
     substantiated" and "word count is 250-300" - this is impossible to
     check reliably against a blob of markdown.
  3. The Streamlit frontend can render nicer widgets (badges for evidence
     type, metrics for marks, etc.) from structured data.
"""
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# Routing (lightweight agent decides RAG vs direct LLM)
# ============================================================
class RouteDecision(BaseModel):
    use_rag: bool = Field(..., description="True if the question needs grounded retrieval from the vector DB")
    reasoning: str = Field(..., description="One-line reason for the routing decision")
    search_query: str = Field(..., description="Best short query to use for vector search if use_rag is True")


# ============================================================
# Condensed retrieval context (Summarizer agent output)
# ============================================================
class CondensedContext(BaseModel):
    summary: str = Field(..., description="Dense, faithful summary of all retrieved material relevant to the question, with no loss of facts/figures/citations")
    key_citations: List[str] = Field(default_factory=list, description="Source file + page references used, e.g. 'laxmikanth_polity.pdf#p42'")
    coverage_gaps: List[str] = Field(default_factory=list, description="Aspects of the question NOT covered by the retrieved material, if any")


# ============================================================
# UPSC Mains answer (point-format body, evidence-tagged)
# ============================================================
EvidenceType = Literal[
    "Constitutional Provision",
    "Supreme Court / High Court Judgment",
    "Committee or Commission Report",
    "Government Scheme or Policy",
    "Statistical Data",
    "Parliamentary Report or Act",
    "Expert Opinion / Other",
]


class BodyPoint(BaseModel):
    point: str = Field(..., description="One clear argument or dimension of the answer")
    evidence: str = Field(..., description="Concrete substantiation: Article/judgment/report/scheme/statistic backing the point. Never empty.")
    evidence_type: EvidenceType = Field(..., description="Category of the evidence cited")


class MainsAnswer(BaseModel):
    introduction: str = Field(..., description="2-3 line catchy, relevant introduction")
    body_points: List[BodyPoint] = Field(..., description="The answer body as discrete, substantiated points - never flowing paragraphs")
    conclusion: str = Field(..., description="2-3 line balanced, forward-looking conclusion")
    word_count: int = Field(..., description="Total word count of introduction + body + conclusion")
    gs_paper: Optional[str] = Field(default=None)
    marks: int = Field(default=15)


# ============================================================
# Groundedness verification (FactChecker / Verifier agent output)
#
# Checks every BodyPoint.evidence claim in a freshly-formatted MainsAnswer
# against the CondensedContext.summary it was supposedly drawn from -
# catching evidence that was fabricated rather than retrieved. See
# app/agents/guardrails.py::groundedness_guardrail, which wires a failure
# here directly into the UPSCFormatter task's existing guardrail-retry loop.
# ============================================================
class ClaimVerification(BaseModel):
    claim: str = Field(..., description="The evidence claim being checked, verbatim from the answer")
    supported: bool = Field(..., description="True only if this exact claim is actually grounded in the provided source material")
    matching_source: str = Field(default="", description="The specific part of the source material that supports this claim, or a short reason it is NOT supported")


class VerificationResult(BaseModel):
    verified_claims: List[ClaimVerification] = Field(default_factory=list)
    all_supported: bool = Field(..., description="True only if every single claim was independently verified as supported - false if even one is unsupported")
    unsupported_claims: List[str] = Field(default_factory=list, description="The claims found unsupported, surfaced verbatim for guardrail retry feedback")


# ============================================================
# UPSC Essay (flowing prose, multidimensional)
# ============================================================
class EssayParagraph(BaseModel):
    dimension: str = Field(..., description="The lens this paragraph addresses, e.g. 'Economic', 'Ethical', 'Global/Geopolitical', 'Historical', 'Technological'")
    content: str = Field(..., description="Flowing prose paragraph (not bullet points) developing this dimension with examples/data")


class EssayAnswer(BaseModel):
    title: str
    introduction: str = Field(..., description="An engaging hook paragraph that sets up the thesis")
    body_paragraphs: List[EssayParagraph] = Field(..., description="4-7 paragraphs, each a distinct dimension, in flowing prose")
    conclusion: str = Field(..., description="A visionary, synthesising conclusion tying back to the thesis")
    word_count: int


# ============================================================
# MCQ generation (Prelims)
# ============================================================
class MCQItem(BaseModel):
    id: int
    stem: str
    options: dict[str, str]
    answer: str
    difficulty: Literal["easy", "medium", "hard"]
    explanation: str
    syllabus_tag: str
    source_ref: Optional[str] = None


class MCQBatch(BaseModel):
    topic: str
    questions: List[MCQItem]
    subject_distribution: Optional[dict[str, int]] = None
    sources: List[str] = Field(default_factory=list)


# ============================================================
# Mains question generation
# ============================================================
class MainsQuestionItem(BaseModel):
    id: int
    question: str
    marks: int
    word_limit: int
    gs_paper: str
    syllabus_tag: str
    model_outline: List[str]
    key_points: List[str]
    recent_examples: List[str] = Field(default_factory=list)


class MainsQuestionBatch(BaseModel):
    topic: str
    questions: List[MainsQuestionItem]
    total_marks: Optional[int] = None
    sources: List[str] = Field(default_factory=list)


# ============================================================
# Notes / summary
# ============================================================
class NotesOutput(BaseModel):
    topic: str
    notes_markdown: str = Field(..., description="Topper-style notes: headers, crisp bullets, no paragraphs over 3 lines")
    mains_angle: List[str] = Field(default_factory=list, description="Likely mains question framings on this topic")
    prelims_facts: List[str] = Field(default_factory=list, description="Crisp, memorisable factual bullets for prelims")
    sources: List[str] = Field(default_factory=list)


# ============================================================
# Evaluators
# ============================================================
class MainsEvaluation(BaseModel):
    content: int = Field(..., ge=0, le=5)
    structure: int = Field(..., ge=0, le=3)
    examples: int = Field(..., ge=0, le=3)
    language: int = Field(..., ge=0, le=2)
    conclusion: int = Field(..., ge=0, le=2)
    total: int = Field(..., ge=0, le=15)
    strengths: List[str]
    weaknesses: List[str]
    missed_points: List[str]
    improvement_tips: List[str]
    improved_intro: str
    improved_conclusion: str
    reference_sources: List[str] = Field(default_factory=list)


class EssayEvaluation(BaseModel):
    content_relevance: int = Field(..., ge=0, le=35)
    multidimensionality: int = Field(..., ge=0, le=25)
    structure_coherence: int = Field(..., ge=0, le=25)
    language_expression: int = Field(..., ge=0, le=25)
    conclusion: int = Field(..., ge=0, le=15)
    total: int = Field(..., ge=0, le=125)
    strengths: List[str]
    weaknesses: List[str]
    improvement_tips: List[str]
    missing_dimensions: List[str] = Field(default_factory=list)


# ============================================================
# PYQ matcher
# ============================================================
class PYQMatchItem(BaseModel):
    year: Optional[int] = None
    exam_type: Optional[str] = Field(default=None, description="Prelims / Mains")
    paper: Optional[str] = Field(default=None, description="e.g. GS-II, CSAT")
    question: str
    note: Optional[str] = Field(default=None, description="Why this PYQ is relevant / how closely it matches")


class PYQMatchResult(BaseModel):
    query_topic: str
    matches: List[PYQMatchItem]
    frequency_insight: str = Field(..., description="Examiner-pattern insight, e.g. 'asked in 4 of the last 10 years, mostly in GS-II'")


# ============================================================
# Syllabus mapper
# ============================================================
class SyllabusMap(BaseModel):
    prelims_topics: List[str] = Field(default_factory=list)
    gs1_topics: List[str] = Field(default_factory=list)
    gs2_topics: List[str] = Field(default_factory=list)
    gs3_topics: List[str] = Field(default_factory=list)
    gs4_topics: List[str] = Field(default_factory=list)
    essay_themes: List[str] = Field(default_factory=list)
    current_affairs_link: str = ""


# ============================================================
# Daily current-affairs digest (generated at ingestion time)
# ============================================================
class DigestOutput(BaseModel):
    source_file: str
    date_tag: str
    headline_summary: str
    prelims_facts: List[str]
    mains_points: List[str]
    gs_paper_tags: List[str]
