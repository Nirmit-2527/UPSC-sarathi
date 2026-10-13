#!/usr/bin/env bash
# =============================================================================
# upsc-sarathi-agentic :: scripts/setup_local.sh
#
# One-command bootstrap for the LOCAL (Ollama + Chroma) profile:
#   1. Creates Python virtualenv
#   2. Installs all dependencies
#   3. Sets up .env if missing
#   4. Creates data directories
#   5. Checks Ollama is running and the required models are pulled
#   6. (Optional) Ingests sample PDFs from ./data/sample_pdfs
#   7. (Optional) Starts FastAPI + Streamlit in the background
#
# Usage:
#   bash scripts/setup_local.sh                  # install + bootstrap (no run)
#   bash scripts/setup_local.sh --ingest          # also ingest sample_pdfs
#   bash scripts/setup_local.sh --run             # also start API + UI
#   bash scripts/setup_local.sh --all             # do everything
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLU}[setup]${NC} $*"; }
ok()   { echo -e "${GRN}[ok]${NC} $*"; }
warn() { echo -e "${YLW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[err]${NC} $*"; exit 1; }

DO_INGEST=0
DO_RUN=0
for arg in "$@"; do
  case "$arg" in
    --ingest) DO_INGEST=1 ;;
    --run)    DO_RUN=1 ;;
    --all)    DO_INGEST=1; DO_RUN=1 ;;
    *) die "Unknown arg: $arg" ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
log "Project root: $ROOT"

# ---------- 1. Python check ----------
PY="${PYTHON:-python3}"
$PY -c "import sys; assert sys.version_info >= (3, 10), sys.version" \
  || die "Python 3.10+ required. Found: $($PY --version)"
ok "Python: $($PY --version)"

# ---------- 2. venv ----------
if [[ ! -d ".venv" ]]; then
  log "Creating virtualenv .venv ..."
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtualenv active: $(which python)"

# ---------- 3. deps ----------
log "Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools >/dev/null
log "Installing requirements (this may take ~3-5 min the first time)..."
python -m pip install -r requirements.txt
ok "Dependencies installed."

# ---------- 4. .env ----------
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  ok "Created .env from template (local/Ollama profile is the default)."
else
  ok ".env already present."
fi

# ---------- 5. data dirs ----------
mkdir -p data/chroma data/sample_pdfs data/sample_pyqs data/sample_current_affairs
ok "Data directories ready."

# ---------- 6. Ollama check ----------
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama not found on PATH. Install it from https://ollama.com/download, then re-run this script."
  warn "(Or run: bash scripts/pull_ollama_models.sh  once it's installed.)"
else
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama is running."
    bash scripts/pull_ollama_models.sh
  else
    warn "Ollama is installed but not running. Start it (run 'ollama serve' or open the Ollama app), then run:"
    warn "  bash scripts/pull_ollama_models.sh"
  fi
fi

# ---------- 7. Optional: ingest sample PDFs ----------
if [[ "$DO_INGEST" -eq 1 ]]; then
  shopt -s nullglob
  pdfs=(data/sample_pdfs/*.pdf)
  if [[ ${#pdfs[@]} -eq 0 ]]; then
    warn "No PDFs in data/sample_pdfs. Add some and re-run with --ingest."
  else
    log "Ingesting ${#pdfs[@]} PDF(s) from data/sample_pdfs ..."
    python -m app.ingestion.pipeline ingest --path data/sample_pdfs
    ok "Ingestion done."
  fi
fi

# ---------- 8. Optional: run servers ----------
if [[ "$DO_RUN" -eq 1 ]]; then
  log "Launching FastAPI on :8000 and Streamlit on :8501 ..."
  mkdir -p logs
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  pkill -f "streamlit run frontend/streamlit_app.py" 2>/dev/null || true
  nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
  sleep 2
  nohup python -m streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true > logs/ui.log 2>&1 &
  sleep 2
  ok "API:        http://localhost:8000/docs"
  ok "Streamlit:  http://localhost:8501"
  ok "Logs:       tail -f logs/api.log  |  tail -f logs/ui.log"
fi

echo
ok "DONE."
echo
cat <<EOF
Next steps:
  1. Make sure Ollama is running with the required models (bash scripts/pull_ollama_models.sh).
  2. Drop UPSC PDFs into ./data/sample_pdfs, ./data/sample_pyqs, ./data/sample_current_affairs and run:
        bash scripts/ingest_docs.sh data/sample_pdfs
        bash scripts/ingest_docs.sh data/sample_pyqs --doc-type pyq
        bash scripts/ingest_docs.sh data/sample_current_affairs --doc-type current_affairs
  3. Start the app:
        bash scripts/setup_local.sh --run
     or manually:
        source .venv/bin/activate
        uvicorn app.main:app --reload &
        streamlit run frontend/streamlit_app.py
  4. Open:
        http://localhost:8000/docs       (API)
        http://localhost:8501            (UI)
EOF
