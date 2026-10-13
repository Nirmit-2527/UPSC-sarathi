#!/usr/bin/env bash
# =============================================================================
# upsc-sarathi-agentic :: scripts/teardown.sh
# Stops local servers and (optionally) destroys AWS resources / wipes local data.
#
# Usage:
#   bash scripts/teardown.sh                 # local stop only
#   bash scripts/teardown.sh --aws           # also destroy AWS infra
#   bash scripts/teardown.sh --wipe-local    # also delete local chroma data
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[teardown] stopping local API + UI ..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "streamlit run frontend/streamlit_app.py" 2>/dev/null || true

for arg in "$@"; do
  case "$arg" in
    --aws)
      bash scripts/deploy_aws.sh destroy
      ;;
    --wipe-local)
      rm -rf data/chroma
      echo "[teardown] removed data/chroma"
      ;;
  esac
done

echo "[teardown] done."
