"""
upsc-sarathi-agentic :: app/ingestion/chunker.py

Recursive character-level chunker tuned for UPSC study material:
  - Try splitting on sub-section markers (CHAPTER, Section, numbered Articles)
  - Fall back to paragraphs / sentences / characters.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


UPSC_SEPARATORS = [
    "\n\nCHAPTER ",
    "\n\nChapter ",
    "\n\nSECTION ",
    "\n\nSection ",
    "\n\nArticle ",
    "\n\nPart ",
    "\n\nSchedule ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: [{"page": int, "text": str, "source": str}, ...]
    Output: [{"text": str, "source": str, "page": int, "chunk_id": str}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=UPSC_SEPARATORS,
        length_function=len,
    )

    chunks: List[Dict[str, Any]] = []
    for p in pages:
        text = _clean(p["text"])
        if not text.strip():
            continue
        for i, ch in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "text": ch,
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_id": f"{p['source']}#p{p['page']}#c{i}",
                }
            )
    return chunks


def _clean(text: str) -> str:
    text = re.sub(r"\r", "", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
