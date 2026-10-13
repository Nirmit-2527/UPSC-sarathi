"""
upsc-sarathi-agentic :: app/ingestion/pyq_parser.py

Previous-Year-Question (PYQ) PDFs are usually a compilation of many
short, numbered questions rather than continuous prose, so the generic
RecursiveCharacterTextSplitter (tuned for textbook paragraphs) produces poor
chunks for them - it might cut a question in half or merge three questions
into one chunk, which hurts the PYQ-matcher agent's semantic search.

This module:
  1. Extracts exam year + exam type (Prelims/Mains) from the filename if the
     aspirant follows a simple naming convention, e.g.:
        Prelims_GS1_2023.pdf, Mains_GS2_2021.pdf, UPSC_Prelims_2019.pdf
     (best-effort; falls back to "unknown" if it can't parse - the chunk is
     still ingested and searchable, just without those metadata filters.)
  2. Splits page text into one chunk per question using a numbered-question
     regex pattern ("1.", "Q.1", "1)" at the start of a line), which is by
     far the most common convention in PYQ compilations.
  3. Falls back to the standard chunker if fewer than 2 question boundaries
     are found on a page (e.g. cover pages, instructions).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.ingestion.chunker import chunk_pages

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_PRELIMS_RE = re.compile(r"prelim", re.IGNORECASE)
_MAINS_RE = re.compile(r"main", re.IGNORECASE)
_GS_PAPER_RE = re.compile(r"GS[\s_-]?([1-4]|I{1,3}|IV)", re.IGNORECASE)

# Matches the start of a new numbered question, e.g. "23. ", "Q.23 ", "23) "
_QUESTION_START_RE = re.compile(r"(?:^|\n)\s*(?:Q\.?\s*)?(\d{1,3})[.)]\s+", re.MULTILINE)


def parse_filename_metadata(filename: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    years = _YEAR_RE.findall(filename)
    if years:
        # findall on a group returns the matched group, not the full match; re-extract properly
        full_years = _YEAR_RE.finditer(filename)
        nums = [int(m.group(0)) for m in full_years]
        if nums:
            meta["exam_year"] = max(nums)  # if multiple years appear, assume the most recent is the exam year
    if _PRELIMS_RE.search(filename):
        meta["exam_type"] = "Prelims"
    elif _MAINS_RE.search(filename):
        meta["exam_type"] = "Mains"
    gs = _GS_PAPER_RE.search(filename)
    if gs:
        meta["gs_paper"] = f"GS-{gs.group(1).upper()}"
    return meta


def split_into_questions(text: str) -> List[str]:
    """Split a page (or document) of PYQ text into individual question strings."""
    starts = [m.start() for m in _QUESTION_START_RE.finditer(text)]
    if len(starts) < 2:
        return []
    starts.append(len(text))
    pieces = []
    for i in range(len(starts) - 1):
        piece = text[starts[i]:starts[i + 1]].strip()
        if len(piece) > 15:  # skip stray numeric matches with no real content
            pieces.append(piece)
    return pieces


def chunk_pyq_pages(pages: List[Dict[str, Any]], filename_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Like chunker.chunk_pages, but tries question-boundary splitting first and
    only falls back to the generic chunker per-page when that fails.
    """
    filename_meta = filename_meta or {}
    chunks: List[Dict[str, Any]] = []
    fallback_pages: List[Dict[str, Any]] = []

    for p in pages:
        questions = split_into_questions(p["text"])
        if not questions:
            fallback_pages.append(p)
            continue
        for i, q_text in enumerate(questions):
            chunks.append(
                {
                    "text": q_text,
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_id": f"{p['source']}#p{p['page']}#q{i}",
                }
            )

    if fallback_pages:
        chunks.extend(chunk_pages(fallback_pages))

    return chunks
