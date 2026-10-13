"""
upsc-sarathi-agentic :: app/ingestion/pipeline.py

End-to-end document ingestion:
   load -> chunk (generic or PYQ-aware) -> tag subject -> embed -> upsert into VDB.

doc_type drives both the chunking strategy and the metadata attached to every
chunk, which is what lets agent tools later filter the vector DB (e.g. the
PYQ-matcher tool only searches doc_type="pyq", the digest agent only reads
today's doc_type="current_affairs" chunks).

CLI usage:
    python -m app.ingestion.pipeline ingest --path data/sample_pdfs
    python -m app.ingestion.pipeline ingest --path data/sample_pyqs --doc-type pyq
    python -m app.ingestion.pipeline ingest --path data/sample_current_affairs --doc-type current_affairs
    python -m app.ingestion.pipeline ingest --s3 s3://bucket/prefix/
    python -m app.ingestion.pipeline stats
    python -m app.ingestion.pipeline delete <filename>
"""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional
import typer
from loguru import logger

from app.config import settings
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.pyq_parser import chunk_pyq_pages, parse_filename_metadata
from app.core.vectorstore import get_vector_store


app = typer.Typer(help="UPSC Sarathi Agentic - ingestion CLI")


# ----------------- subject heuristic -----------------
SUBJECT_KEYWORDS = {
    "Polity": ["Constitution", "Article", "Lok Sabha", "Rajya Sabha", "Fundamental Rights",
               "DPSP", "Supreme Court", "Parliament", "Election Commission", "Federal"],
    "History": ["Mughal", "British", "Independence", "Gandhi", "Nehru", "Mauryan",
                "Harappa", "Indus Valley", "Bhakti", "Sufi", "Revolt of 1857"],
    "Geography": ["Monsoon", "Himalaya", "Plateau", "Tropical", "Latitude", "Longitude",
                  "Erosion", "Igneous", "Tectonic", "Drainage"],
    "Economy": ["GDP", "Inflation", "Fiscal", "RBI", "Monetary Policy", "Repo",
                "Budget", "Trade", "Export", "Subsidy", "NITI Aayog"],
    "Environment": ["Biodiversity", "Climate", "Carbon", "Wildlife", "Forest",
                    "Pollution", "IPCC", "UNFCCC", "Tiger Reserve", "Sanctuary"],
    "Science": ["DNA", "Protein", "Nano", "Satellite", "ISRO", "Vaccine", "Quantum",
                "Antibody", "Genome", "AI", "Machine Learning"],
    "Ethics": ["Integrity", "Probity", "Ethics", "Aptitude", "Emotional Intelligence",
               "Civil Service", "Public Service Values"],
    "IR": ["Foreign Policy", "Treaty", "UN", "WTO", "BRICS", "Quad", "G20",
           "Bilateral", "Multilateral"],
}


def tag_subject(text: str) -> str:
    scores = {sub: 0 for sub in SUBJECT_KEYWORDS}
    lowered = text.lower()
    for sub, kws in SUBJECT_KEYWORDS.items():
        for k in kws:
            if k.lower() in lowered:
                scores[sub] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "General"


# ----------------- core ingest -----------------
def ingest_files(
    paths: List[Path],
    doc_type: str = "study_material",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    doc_type: "study_material" | "pyq" | "current_affairs"
    extra_meta: merged into every chunk's metadata (e.g. {"batch": "2025-06-17"})
    """
    vs = get_vector_store()
    total_chunks = 0
    per_file: List[Dict[str, Any]] = []
    today = date.today().isoformat()

    for p in paths:
        if not p.exists():
            logger.warning(f"Skipping missing file: {p}")
            continue
        if p.suffix.lower() not in (".pdf",):
            logger.warning(f"Skipping non-PDF: {p.name}")
            continue
        try:
            pages = load_pdf(p)
            if not pages:
                logger.warning(f"No text extracted from {p.name}")
                continue

            filename_meta = parse_filename_metadata(p.name) if doc_type == "pyq" else {}

            if doc_type == "pyq":
                chunks = chunk_pyq_pages(pages, filename_meta)
            else:
                chunks = chunk_pages(pages)

            if not chunks:
                logger.warning(f"No chunks produced from {p.name}")
                continue

            texts = [c["text"] for c in chunks]
            metas: List[Dict[str, Any]] = []
            for c in chunks:
                meta: Dict[str, Any] = {
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_id": c["chunk_id"],
                    "subject": tag_subject(c["text"]),
                    "doc_type": doc_type,
                    "ingested_on": today,
                }
                if doc_type == "pyq":
                    meta["exam_year"] = filename_meta.get("exam_year", 0)
                    meta["exam_type"] = filename_meta.get("exam_type", "unknown")
                    if "gs_paper" in filename_meta:
                        meta["gs_paper"] = filename_meta["gs_paper"]
                if extra_meta:
                    meta.update(extra_meta)
                metas.append(meta)

            ids = vs.add(texts, metas)
            total_chunks += len(ids)
            per_file.append({
                "file": p.name, "pages": len(pages), "chunks": len(chunks),
                "doc_type": doc_type, **filename_meta,
            })
            logger.success(f"Ingested {p.name}: {len(pages)} pages, {len(chunks)} chunks, doc_type={doc_type}")
        except Exception as e:
            logger.exception(f"Failed to ingest {p.name}: {e}")

    return {"total_chunks_added": total_chunks, "files": per_file, "store_size": vs.count()}


# ----------------- S3 ingest (AWS profile) -----------------
def _download_s3_prefix(s3_uri: str, dst: Path) -> List[Path]:
    import boto3

    assert s3_uri.startswith("s3://"), "Expected s3://bucket/prefix"
    bucket, _, prefix = s3_uri[5:].partition("/")
    s3 = boto3.client("s3", region_name=settings.aws_region)
    paginator = s3.get_paginator("list_objects_v2")
    out: List[Path] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".pdf"):
                continue
            local = dst / Path(key).name
            s3.download_file(bucket, key, str(local))
            out.append(local)
            logger.info(f"Downloaded s3://{bucket}/{key} -> {local}")
    return out


# ----------------- CLI -----------------
@app.command()
def ingest(
    path: Optional[Path] = typer.Option(None, help="Local directory or file"),
    s3: Optional[str] = typer.Option(None, help="s3://bucket/prefix to ingest from"),
    doc_type: str = typer.Option("study_material", help="study_material | pyq | current_affairs"),
):
    """Ingest PDFs from a local path or S3 prefix into the vector store."""
    files: List[Path] = []
    if path:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.pdf")))
        else:
            typer.echo(f"Path not found: {path}", err=True)
            raise typer.Exit(2)
    if s3:
        with tempfile.TemporaryDirectory() as td:
            files.extend(_download_s3_prefix(s3, Path(td)))
            result = ingest_files(files, doc_type=doc_type)
            typer.echo(result)
            return
    if not files:
        typer.echo("Provide --path or --s3", err=True)
        raise typer.Exit(2)
    result = ingest_files(files, doc_type=doc_type)
    typer.echo(result)


@app.command()
def stats():
    """Show vector store stats."""
    vs = get_vector_store()
    typer.echo({"vector_store": settings.vector_store, "count": vs.count()})


@app.command()
def delete(source: str = typer.Argument(..., help="filename to delete from VDB")):
    """Remove all chunks belonging to a given source filename."""
    vs = get_vector_store()
    n = vs.delete_by_source(source)
    typer.echo({"deleted": n, "source": source})


if __name__ == "__main__":
    app()
