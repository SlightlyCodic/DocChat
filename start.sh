#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== RAG Document Chatbot ==="
echo ""

# Check Ollama
printf "Checking Ollama... "
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "NOT RUNNING"
  echo ""
  echo "  Start Ollama first:   ollama serve"
  echo "  Then re-run:          ./start.sh"
  exit 1
fi
echo "OK"

# Ensure model exists
printf "Checking llama3.2... "
if ! ollama list | grep -q "llama3.2"; then
  echo "not found, pulling..."
  ollama pull llama3.2
else
  echo "OK"
fi

echo ""
echo "Installing Python dependencies..."
pip3 install -q -r "$DIR/backend/requirements.txt"
echo "Done."
echo ""
echo "Server starting at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

cd "$DIR/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
