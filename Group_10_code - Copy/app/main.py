"""
upsc-sarathi-agentic :: app/main.py
FastAPI entrypoint.
Run:    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.api.routes.api import router as api_router


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}")


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(
        title="UPSC Sarathi Agentic",
        version="1.0.0",
        description=(
            "Agentic RAG UPSC Civil Services tutor, built on CrewAI. A CrewAI router agent decides "
            "direct-LLM vs RAG per question; specialist agent crews handle Q&A, mains-answer writing "
            "(strict point-format + evidence + word-count guardrails), essay writing, MCQ/mains question "
            "generation, topper notes, mains/essay evaluation, PYQ matching, current-affairs digesting, "
            "and syllabus mapping. Runs on Ollama+Chroma locally or Bedrock+OpenSearch Serverless on AWS, "
            "switched purely via environment variables."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    def root():
        return {
            "name": "UPSC Sarathi Agentic",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "mode": settings.app_mode,
            "vector_store": settings.vector_store,
            "llm_provider": settings.llm_provider,
        }

    logger.info(
        f"App ready. Mode={settings.app_mode}, VDB={settings.vector_store}, "
        f"LLM={settings.llm_provider}, Embeddings={settings.embeddings_provider}"
    )
    return app


app = create_app()
