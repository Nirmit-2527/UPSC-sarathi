"""
upsc-sarathi-agentic :: app/agents/tools.py

CrewAI tools (subclasses of crewai.tools.BaseTool) that wrap the plain,
deterministic `Retriever` (app/core/retrieval.py) so agents can call vector
search themselves instead of the orchestration layer always pre-fetching
context for them.

Two tools:
  - VectorSearchTool: general-purpose search over study material / notes /
    current-affairs digests, optionally scoped by subject or doc_type.
  - PYQSearchTool: search scoped to doc_type="pyq" only, used by the PYQ
    Matcher agent (feature #6 - "last 10 years of previous year questions").

Both tools return a plain string (CrewAI tools must return text the LLM can
read) built via `Retriever.build_context()`, plus a compact source list the
calling agent is instructed (via its task description) to copy into
`sources` / `key_citations` fields of its structured output.
"""
from __future__ import annotations

from typing import Optional, Type, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from app.core.retrieval import get_retriever


class VectorSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    k: int = Field(default=6, description="Number of chunks to retrieve")
    subject: Optional[str] = Field(
        default=None,
        description="Optional subject filter, e.g. Polity, Economy, Geography, History, Environment, Science, Ethics, InternationalRelations, CurrentAffairs",
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Optional doc type filter: study_material | pyq | current_affairs | digest",
    )


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = (
        "Search the ingested UPSC vector database (study material, notes, "
        "current-affairs digests) for passages relevant to a query. Returns "
        "the matched passages with their source file and page number. "
        "Always cite the [source p.X] tags shown in the result when you use "
        "a fact."
    )
    args_schema: Type[BaseModel] = VectorSearchInput

    def _run(self, query: str, k: int = 6, subject: Optional[str] = None, doc_type: Optional[str] = None) -> str:
        retriever = get_retriever()
        filt = {}
        if subject:
            filt["subject"] = subject
        if doc_type:
            filt["doc_type"] = doc_type
        chunks = retriever.retrieve(query, k=k, filter=filt or None)
        if not chunks:
            return "No relevant material found in the vector database for this query."
        return retriever.build_context(chunks)


class PYQSearchInput(BaseModel):
    topic: str = Field(..., description="Topic / keyword to match against previous year questions")
    k: int = Field(default=10, description="Number of PYQ chunks to retrieve")
    exam_type: Optional[str] = Field(default=None, description="Optional filter: Prelims or Mains")


class PYQSearchTool(BaseTool):
    name: str = "pyq_search"
    description: str = (
        "Search ONLY the ingested Previous Year Question (PYQ) bank for "
        "questions related to a topic. Use this to find how often and in "
        "what form a topic has been asked in the last ~10 years of UPSC "
        "exams. Returns each matched PYQ with its source file/page (the "
        "filename often encodes the exam year and type, e.g. "
        "'Mains_GS2_2021.pdf')."
    )
    args_schema: Type[BaseModel] = PYQSearchInput

    def _run(self, topic: str, k: int = 10, exam_type: Optional[str] = None) -> str:
        retriever = get_retriever()
        filt = {"doc_type": "pyq"}
        if exam_type:
            filt["exam_type"] = exam_type
        chunks = retriever.retrieve(topic, k=k, filter=filt)
        if not chunks:
            return "No matching previous year questions found in the ingested PYQ bank."
        return retriever.build_context(chunks, max_chars=6000)
