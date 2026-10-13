"""
upsc-sarathi-agentic :: app/api/routes/api.py

All HTTP endpoints in one router. Carries forward every endpoint from the
non-agentic reference project, and adds the agentic-only extensions:
    /mains/write            - generate a model UPSC mains answer on demand
    /essay/write            - generate a ~1500-word essay
    /evaluate/essay         - evaluate an essay out of 125
    /pyq/match              - standalone PYQ matcher (feature #6)
    /ingest/current-affairs - dedicated current-affairs ingestion (+ digest)
    /ingest/pyq             - dedicated PYQ ingestion (per-question chunking)
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.config import settings
from app.models import schemas
from app.services import services
from app.services import revision
from app.core.vectorstore import get_vector_store


router = APIRouter()


def _save_uploads(files: List[UploadFile]) -> tuple[Path, List[Path]]:
    tmp = Path(tempfile.mkdtemp())
    saved: List[Path] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        p = tmp / f.filename
        with p.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(p)
    return tmp, saved


# ---------- health ----------
@router.get("/health", tags=["meta"])
def health():
    vs = get_vector_store()
    return {"status": "ok", "vector_store_count": vs.count(), "llm_provider": settings.llm_provider}


@router.get("/stats", tags=["meta"])
def stats():
    return {"vector_store_count": get_vector_store().count()}


# ---------- Q&A (agentic: router -> direct | RAG, optional PYQ) ----------
@router.post("/qa", response_model=schemas.QAResponse, tags=["features"])
def qa(req: schemas.QARequest):
    try:
        result = services.qa(req.question, k=req.k, subject=req.subject, include_pyq=req.include_pyq)
        return schemas.QAResponse(
            answer=result["answer"],
            sources=[schemas.Source(**s) for s in result["sources"]],
            used_rag=result["used_rag"],
            route_reason=result["route_reason"],
            agent_trace=result["agent_trace"],
        )
    except Exception as e:
        logger.exception("qa failed")
        raise HTTPException(500, str(e))


# ---------- Mains answer writer (model answer on demand) ----------
@router.post("/mains/write", tags=["features"])
def mains_write(req: schemas.MainsWriteRequest):
    try:
        result = services.mains_write(req.question, marks=req.marks, gs_paper=req.gs_paper)
        return {
            "mains_answer": result["mains_answer"],
            "pyq_matches": result["pyq_matches"],
            "agent_trace": result["agent_trace"],
            "groundedness_checked": result.get("groundedness_checked", False),
        }
    except Exception as e:
        logger.exception("mains_write failed")
        raise HTTPException(500, str(e))


# ---------- Essay writer ----------
@router.post("/essay/write", tags=["features"])
def essay_write(req: schemas.EssayWriteRequest):
    try:
        result = services.essay_write(req.topic)
        return {"essay": result["essay"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("essay_write failed")
        raise HTTPException(500, str(e))


# ---------- MCQ generation ----------
@router.post("/generate/mcq", tags=["features"])
def gen_mcq(req: schemas.MCQRequest):
    try:
        result = services.generate_mcq(req.topic, n=req.n, subject=req.subject)
        return {"mcq_batch": result["mcq_batch"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("gen_mcq failed")
        raise HTTPException(500, str(e))


# ---------- Mains question generation ----------
@router.post("/generate/mains", tags=["features"])
def gen_mains(req: schemas.MainsRequest):
    try:
        result = services.generate_mains_questions(req.topic, n=req.n, gs_paper=req.gs_paper)
        return {"mains_question_batch": result["mains_question_batch"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("gen_mains failed")
        raise HTTPException(500, str(e))


# ---------- Notes ----------
@router.post("/generate/notes", tags=["features"])
def gen_notes(req: schemas.NotesRequest):
    try:
        result = services.generate_notes(req.topic, subject=req.subject)
        return {"notes": result["notes"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("gen_notes failed")
        raise HTTPException(500, str(e))


# ---------- Mock tests ----------
@router.post("/test/prelims", tags=["tests"])
def prelims_paper(req: schemas.PrelimsPaperRequest):
    try:
        result = services.generate_prelims_paper(req.topics, n=req.n)
        return {"mcq_batch": result["mcq_batch"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("prelims_paper failed")
        raise HTTPException(500, str(e))


@router.post("/test/mains", tags=["tests"])
def mains_paper(req: schemas.MainsPaperRequest):
    try:
        result = services.generate_mains_paper(req.topic, gs_paper=req.gs_paper, n=req.n)
        return {"mains_question_batch": result["mains_question_batch"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("mains_paper failed")
        raise HTTPException(500, str(e))


# ---------- Evaluators ----------
@router.post("/evaluate", tags=["features"])
def evaluate(req: schemas.AnswerEvalRequest):
    try:
        result = services.evaluate_mains_answer(req.question, req.answer)
        return {"evaluation": result["evaluation"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("evaluate failed")
        raise HTTPException(500, str(e))


@router.post("/evaluate/essay", tags=["features"])
def evaluate_essay(req: schemas.EssayEvalRequest):
    try:
        result = services.evaluate_essay(req.topic, req.essay_text)
        return {"evaluation": result["evaluation"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("evaluate_essay failed")
        raise HTTPException(500, str(e))


# ---------- PYQ matcher (standalone) ----------
@router.post("/pyq/match", tags=["features"])
def pyq_match(req: schemas.PYQMatchRequest):
    try:
        result = services.pyq_match(req.topic, k=req.k)
        return {"pyq_result": result["pyq_result"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("pyq_match failed")
        raise HTTPException(500, str(e))


# ---------- Syllabus mapper ----------
@router.post("/syllabus/map", tags=["features"])
def syllabus(req: schemas.SyllabusMapRequest):
    try:
        result = services.syllabus_map(req.text)
        return {"syllabus_map": result["syllabus_map"], "agent_trace": result["agent_trace"]}
    except Exception as e:
        logger.exception("syllabus failed")
        raise HTTPException(500, str(e))


# ---------- Ingestion ----------
@router.post("/ingest/upload", tags=["ingest"])
async def upload(files: List[UploadFile] = File(...)):
    """Upload one or more study-material PDFs; ingested as doc_type=study_material."""
    tmp, saved = _save_uploads(files)
    try:
        if not saved:
            raise HTTPException(400, "No PDFs received")
        return services.ingest_study_material(saved)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/ingest/pyq", tags=["ingest"])
async def upload_pyq(files: List[UploadFile] = File(...)):
    """Upload Previous Year Question compilation PDFs; chunked per-question
    and tagged doc_type=pyq, exam_year/exam_type/gs_paper parsed from filename
    where possible (feature #6)."""
    tmp, saved = _save_uploads(files)
    try:
        if not saved:
            raise HTTPException(400, "No PDFs received")
        return services.ingest_pyq(saved)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/ingest/current-affairs", tags=["ingest"])
async def upload_current_affairs(files: List[UploadFile] = File(...)):
    """Upload daily current-affairs PDFs; tagged doc_type=current_affairs +
    today's date, and (if enabled) summarised by the DigestCurator agent
    into a compact digest stored back into the vector DB (feature #5)."""
    tmp, saved = _save_uploads(files)
    try:
        if not saved:
            raise HTTPException(400, "No PDFs received")
        return services.ingest_current_affairs(saved)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- Spaced repetition ----------
@router.post("/sr/add", tags=["revision"])
def sr_add(req: schemas.SRAddRequest):
    try:
        c = revision.add_card(req.topic, req.front, req.back)
        return c.__dict__
    except Exception as e:
        logger.exception("sr_add failed")
        raise HTTPException(500, str(e))


@router.post("/sr/review", tags=["revision"])
def sr_review(req: schemas.SRReviewRequest):
    try:
        c = revision.review(req.card_id, req.quality)
        return c.__dict__
    except KeyError:
        raise HTTPException(404, f"No such revision card: {req.card_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("sr_review failed")
        raise HTTPException(500, str(e))


@router.get("/sr/due", tags=["revision"])
def sr_due(limit: int = 20):
    try:
        return [c.__dict__ for c in revision.due_cards(limit=limit)]
    except Exception as e:
        logger.exception("sr_due failed")
        raise HTTPException(500, str(e))
