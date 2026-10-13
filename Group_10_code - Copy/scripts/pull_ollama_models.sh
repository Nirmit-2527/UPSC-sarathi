#!/usr/bin/env bash
# =============================================================================
# upsc-sarathi-agentic :: scripts/pull_ollama_models.sh
#
# Pulls every model the local profile needs:
#   - llama3.1:8b     -> "smart" tier (writing/planning/evaluating/formatting)
#   - llama3.2:3b     -> "fast" tier (routing/PYQ matching/tagging)
#   - nomic-embed-text -> embeddings
#
# Safe to re-run; `ollama pull` is a no-op if the model is already present.
# =============================================================================
set -euo pipefail

command -v ollama >/dev/null 2>&1 || { echo "ollama not found on PATH - install from https://ollama.com/download"; exit 1; }

for m in llama3.1:8b llama3.2:3b nomic-embed-text; do
  echo "[pull] $m"
  ollama pull "$m"
done

echo "[ok] All local models ready."
