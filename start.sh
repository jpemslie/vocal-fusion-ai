#!/usr/bin/env bash
# VocalFusion — quick local start (no Docker)
# Starts FastAPI backend + Next.js UI dev server side by side.
#
# Usage:  bash start.sh

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Use conda Node.js if system node not found
if ! command -v node &>/dev/null; then
  CONDA_NODE="/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin"
  export PATH="$CONDA_NODE:$PATH"
fi

echo "==> Starting VocalFusion API on :8000"
cd "$ROOT"
uvicorn api_fast:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "==> Starting VocalFusion UI on :3000"
cd "$ROOT/ui"
BACKEND_URL="http://localhost:8000" npm run dev &
UI_PID=$!

echo ""
echo "  API : http://localhost:8000/health"
echo "  UI  : http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both."

trap "echo; echo 'Stopping...'; kill $API_PID $UI_PID 2>/dev/null; wait" INT TERM
wait
