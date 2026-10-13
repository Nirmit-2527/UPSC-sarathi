"""
upsc-sarathi-agentic :: app/utils/word_count.py

Small, dependency-free helpers shared by app/agents/guardrails.py.

These exist mainly so that the *guardrail itself* (a plain Python callable,
per CrewAI's `Task(guardrail=...)` mechanism) can count words against a
structured Pydantic object rather than a markdown blob - which is what lets
us actually enforce "250-300 words, every point substantiated" instead of
just asking nicely in a prompt.
"""
from __future__ import annotations

import re
from typing import Tuple

from app.config import settings
from app.models.agent_schemas import MainsAnswer, EssayAnswer


def count_words(text: str) -> int:
    """Whitespace word count, with punctuation-only tokens stripped out."""
    if not text:
        return 0
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    return len(tokens)


def marks_to_word_range(marks: int) -> Tuple[int, int]:
    """UPSC mains convention: roughly ~13-15 words per mark for short
    answers, tapering for the 10/15/20-mark question types. The 15-mark
    bucket below intentionally matches the user's explicit "~250-300 words"
    requirement.
    """
    if marks <= 10:
        return (100, 250)
    if marks <= 15:
        return (180, 300)
    if marks <= 20:
        return (300, 400)
    return (240, 300)  # sensible default fallback


def essay_word_range() -> Tuple[int, int]:
    target = settings.essay_target_words
    tol = settings.essay_word_tolerance
    return (max(0, target - tol), target + tol)


def mains_answer_text(answer: MainsAnswer) -> str:
    """Flatten a MainsAnswer into the text a human examiner would actually
    read, for word counting and for plain-text display/export."""
    body = "\n".join(f"- {bp.point} [{bp.evidence_type}: {bp.evidence}]" for bp in answer.body_points)
    return f"{answer.introduction}\n\n{body}\n\n{answer.conclusion}"


def essay_text(essay: EssayAnswer) -> str:
    body = "\n\n".join(p.content for p in essay.body_paragraphs)
    return f"{essay.title}\n\n{essay.introduction}\n\n{body}\n\n{essay.conclusion}"
