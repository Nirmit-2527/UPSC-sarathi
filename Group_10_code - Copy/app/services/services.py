"""
upsc-sarathi-agentic :: app/services/services.py

Thin orchestration layer between the FastAPI routes and the CrewAI crews.
Each function here:
  1. kicks off the relevant crew(s) (app/agents/crews/*.py),
  2. does any small bit of plain-Python post-processing the API/UI needs
     (e.g. building a displayable `Source` list, triggering the digest
     crew on current-affairs ingestion),
  3. returns plain dicts (NOT FastAPI response models - that mapping
     happens in app/api/routes/api.py, keeping this module reusable from
     both the API and the Streamlit app/tests without a FastAPI dependency).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import date
from loguru import logger

from app.config import settings
from app.core.retrieval import get_retriever
from app.ingestion.pipeline import ingest_files
from app.ingestion.pdf_loader import load_pdf

from app.agents.crews.qa_crew import run_qa as _run_qa
from app.agents.crews.mains_answer_crew import run_mains_answer as _run_mains_answer
from app.agents.crews.essay_crew import run_essay as _run_essay
from app.agents.crews.question_gen_crew import (
    run_mcq_batch as _run_mcq_batch,
    run_mains_question_batch as _run_mains_question_batch,
    run_prelims_paper as _run_prelims_paper,
    run_mains_paper as _run_mains_paper,
)
from app.agents.crews.notes_crew import run_notes as _run_notes
from app.agents.crews.evaluator_crew import (
    run_mains_evaluation as _run_mains_evaluation,
    run_essay_evaluation as _run_essay_evaluation,
)
from app.agents.crews.pyq_matcher_crew import run_pyq_match as _run_pyq_match
from app.agents.crews.digest_crew import run_digest as _run_digest
from app.agents.crews.syllabus_crew import run_syllabus_map as _run_syllabus_map


# ============================================================
# QA (feature: ask a question -> route -> RAG/direct -> answer)
# ============================================================
def qa(question: str, k: int = 6, subject: Optional[str] = None, include_pyq: bool = True) -> Dict[str, Any]:
    result = _run_qa(question, k=k, subject=subject, include_pyq=include_pyq)

    sources: List[Dict[str, Any]] = []
    if result["used_rag"]:
        # A cheap, deterministic, non-LLM retrieval purely for displaying
        # source snippets to the user - independent of whatever the agent
        # itself searched for internally via its tool calls.
        retriever = get_retriever()
        filt = {"subject": subject} if subject else None
        chunks = retriever.retrieve(question, k=k, filter=filt)
        sources = [
            {
                "source": c.metadata.get("source", "?"),
                "page": int(c.metadata.get("page", 0) or 0),
                "score": c.score,
                "snippet": c.short(),
            }
            for c in chunks
        ]

    return {
        "answer": result["answer"],
        "sources": sources,
        "used_rag": result["used_rag"],
        "route_reason": result["route_reason"],
        "agent_trace": result["agent_trace"],
        "pyq_matches": result["pyq_matches"],
    }


# ============================================================
# Mains answer writer (model answer on demand)
# ============================================================
def mains_write(question: str, marks: int = 15, gs_paper: Optional[str] = None) -> Dict[str, Any]:
    return _run_mains_answer(question, marks=marks, gs_paper=gs_paper, include_pyq=True)


# ============================================================
# Essay writer
# ============================================================
def essay_write(topic: str) -> Dict[str, Any]:
    return _run_essay(topic)


# ============================================================
# Question generation
# ============================================================
def generate_mcq(topic: str, n: int = 10, subject: Optional[str] = None) -> Dict[str, Any]:
    return _run_mcq_batch(topic, n=n, subject=subject)


def generate_mains_questions(topic: str, n: int = 5, gs_paper: Optional[str] = None) -> Dict[str, Any]:
    return _run_mains_question_batch(topic, n=n, gs_paper=gs_paper)


def generate_prelims_paper(topics: List[str], n: int = 20) -> Dict[str, Any]:
    return _run_prelims_paper(topics, n=n)


def generate_mains_paper(topic: str, gs_paper: str = "GS-II", n: int = 5) -> Dict[str, Any]:
    return _run_mains_paper(topic, gs_paper=gs_paper, n=n)


# ============================================================
# Notes
# ============================================================
def generate_notes(topic: str, subject: Optional[str] = None) -> Dict[str, Any]:
    return _run_notes(topic, subject=subject)


# ============================================================
# Evaluation
# ============================================================
def evaluate_mains_answer(question: str, answer_text: str) -> Dict[str, Any]:
    return _run_mains_evaluation(question, answer_text)


def evaluate_essay(topic: str, essay_text: str) -> Dict[str, Any]:
    return _run_essay_evaluation(topic, essay_text)


# ============================================================
# PYQ matcher (standalone)
# ============================================================
def pyq_match(topic: str, k: int = 10) -> Dict[str, Any]:
    return _run_pyq_match(topic, k=k)


# ============================================================
# Syllabus mapper
# ============================================================
def syllabus_map(text: str) -> Dict[str, Any]:
    return _run_syllabus_map(text)


# ============================================================
# Ingestion (study material / PYQ / current affairs)
# ============================================================
def ingest_study_material(paths: List[Path]) -> Dict[str, Any]:
    return ingest_files(paths, doc_type="study_material")


def ingest_pyq(paths: List[Path]) -> Dict[str, Any]:
    return ingest_files(paths, doc_type="pyq")


def ingest_current_affairs(paths: List[Path]) -> Dict[str, Any]:
    """Ingest current-affairs PDFs into the vector DB (doc_type=current_affairs,
    tagged with today's date), and - if enabled - run the DigestCurator agent
    on each file and store the resulting compact digest back into the
    vector DB as doc_type=digest, for fast future retrieval (feature #5)."""
    today = date.today().isoformat()
    result = ingest_files(paths, doc_type="current_affairs")

    digests: List[Dict[str, Any]] = []
    if settings.enable_current_affairs_digest:
        vs_input_texts: List[str] = []
        vs_input_metas: List[Dict[str, Any]] = []
        for p in paths:
            if not p.exists() or p.suffix.lower() != ".pdf":
                continue
            try:
                pages = load_pdf(p)
                raw_text = "\n\n".join(pg["text"] for pg in pages)
                digest_result = _run_digest(p.name, today, raw_text)
                digest = digest_result["digest"]
                digests.append({"source_file": p.name, "digest": digest})

                digest_text = (
                    f"{digest.headline_summary}\n\n"
                    f"Prelims facts: {'; '.join(digest.prelims_facts)}\n\n"
                    f"Mains points: {'; '.join(digest.mains_points)}"
                )
                vs_input_texts.append(digest_text)
                vs_input_metas.append({
                    "source": p.name,
                    "page": 0,
                    "chunk_id": f"{p.name}#digest",
                    "subject": "CurrentAffairs",
                    "doc_type": "digest",
                    "ingested_on": today,
                    "gs_paper_tags": ",".join(digest.gs_paper_tags),
                })
            except Exception as e:
                logger.warning(f"Digest generation failed for {p.name}, skipping its digest (raw chunks were still ingested): {e}")
                continue

        if vs_input_texts:
            from app.core.vectorstore import get_vector_store
            get_vector_store().add(vs_input_texts, vs_input_metas)

    result["digests"] = digests
    return result
