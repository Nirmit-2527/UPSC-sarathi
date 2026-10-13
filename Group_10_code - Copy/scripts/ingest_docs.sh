#!/usr/bin/env bash
# =============================================================================
# upsc-sarathi-agentic :: scripts/ingest_docs.sh
#
# Ingest UPSC PDFs into the vector store.
#
# Usage:
#   bash scripts/ingest_docs.sh                                    # ingest data/sample_pdfs/ as study_material
#   bash scripts/ingest_docs.sh /path/to/pdfs                      # local dir / file, study_material
#   bash scripts/ingest_docs.sh /path/to/pyqs --doc-type pyq       # per-question chunking + filename metadata
#   bash scripts/ingest_docs.sh /path/to/ca --doc-type current_affairs
#   bash scripts/ingest_docs.sh s3://my-bucket/raw/                # from S3 prefix (AWS profile)
#   bash scripts/ingest_docs.sh --stats                            # show VDB stats
#   bash scripts/ingest_docs.sh --delete filename.pdf              # remove a source
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then source .venv/bin/activate; fi

TARGET="${1:-data/sample_pdfs}"
DOC_TYPE="study_material"

# parse a trailing --doc-type flag, if present
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--doc-type" ]]; then
    j=$((i + 1))
    DOC_TYPE="${!j:-study_material}"
  fi
done

case "$TARGET" in
  --stats)
    python -m app.ingestion.pipeline stats
    ;;
  --delete)
    [[ -z "${2:-}" ]] && { echo "usage: --delete <source_filename>"; exit 2; }
    python -m app.ingestion.pipeline delete "$2"
    ;;
  s3://*)
    python -m app.ingestion.pipeline ingest --s3 "$TARGET" --doc-type "$DOC_TYPE"
    ;;
  *)
    python -m app.ingestion.pipeline ingest --path "$TARGET" --doc-type "$DOC_TYPE"
    ;;
esac
