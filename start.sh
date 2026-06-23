#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== RAG Document Chatbot (Claude-powered) ==="
echo ""
echo "Installing Python dependencies..."
pip3 install -q -r "$DIR/backend/requirements.txt"
echo "Done."
echo ""
echo "NOTE: FastEmbed will download the embedding model (~90 MB) on first run."
echo ""
echo "Server starting at http://localhost:8000"
echo "Open the app, click 'API Key', and enter your Anthropic API key."
echo "Press Ctrl+C to stop."
echo ""

cd "$DIR/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
