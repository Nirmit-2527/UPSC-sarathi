"""
upsc-sarathi-agentic :: app/ingestion/pdf_loader.py

Multi-backend PDF text extractor.
  - pdfplumber  : best balance of speed + layout/table fidelity (default)
  - pypdf       : fastest, basic
  - unstructured: heavier, sometimes better on irregular layouts
  - textract    : AWS Textract OCR, for scanned/handwritten PDFs (AWS profile only)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from app.config import settings


def load_pdf(path: str | Path) -> List[Dict[str, Any]]:
    """Returns a list of {page, text, source} dicts."""
    path = Path(path)
    backend = "textract" if settings.use_textract else settings.pdf_loader
    logger.info(f"Loading {path.name} with backend={backend}")
    if backend == "pdfplumber":
        return _load_pdfplumber(path)
    if backend == "pypdf":
        return _load_pypdf(path)
    if backend == "unstructured":
        return _load_unstructured(path)
    if backend == "textract":
        return _load_textract(path)
    raise ValueError(f"Unknown PDF backend: {backend}")


# ----------------- pdfplumber -----------------
def _load_pdfplumber(path: Path) -> List[Dict[str, Any]]:
    import pdfplumber

    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            for t in tables:
                rows = ["\t".join((c or "") for c in row) for row in t]
                text += "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]"
            if text.strip():
                pages.append({"page": i, "text": text, "source": path.name})
    return pages


# ----------------- pypdf -----------------
def _load_pypdf(path: Path) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out = []
    for i, p in enumerate(reader.pages, start=1):
        text = p.extract_text() or ""
        if text.strip():
            out.append({"page": i, "text": text, "source": path.name})
    return out


# ----------------- unstructured -----------------
def _load_unstructured(path: Path) -> List[Dict[str, Any]]:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=str(path), strategy="fast")
    grouped: Dict[int, List[str]] = {}
    for el in elements:
        page = getattr(el.metadata, "page_number", 1) or 1
        grouped.setdefault(page, []).append(str(el))
    return [
        {"page": pg, "text": "\n".join(txts), "source": path.name}
        for pg, txts in sorted(grouped.items())
        if "".join(txts).strip()
    ]


# ----------------- AWS Textract (scanned PDFs, AWS profile only) -----------------
def _load_textract(path: Path) -> List[Dict[str, Any]]:
    import boto3

    s3 = boto3.client("s3", region_name=settings.aws_region)
    textract = boto3.client("textract", region_name=settings.aws_region)

    key = f"textract-temp/{path.name}"
    bucket = settings.s3_bucket_name
    logger.info(f"Uploading {path.name} -> s3://{bucket}/{key}")
    s3.upload_file(str(path), bucket, key)

    job = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = job["JobId"]
    logger.info(f"Textract jobId={job_id}; polling...")

    while True:
        time.sleep(3)
        r = textract.get_document_text_detection(JobId=job_id, MaxResults=1)
        status = r["JobStatus"]
        if status == "SUCCEEDED":
            break
        if status == "FAILED":
            raise RuntimeError("Textract job failed")
        logger.debug(f"Textract status={status}")

    pages: Dict[int, List[str]] = {}
    next_token = None
    while True:
        kwargs = {"JobId": job_id, "MaxResults": 1000}
        if next_token:
            kwargs["NextToken"] = next_token
        r = textract.get_document_text_detection(**kwargs)
        for block in r.get("Blocks", []):
            if block["BlockType"] == "LINE":
                pg = block.get("Page", 1)
                pages.setdefault(pg, []).append(block.get("Text", ""))
        next_token = r.get("NextToken")
        if not next_token:
            break

    return [
        {"page": pg, "text": "\n".join(lines), "source": path.name}
        for pg, lines in sorted(pages.items())
    ]
