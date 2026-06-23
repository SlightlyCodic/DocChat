# DocChat — RAG-Powered Document Chatbot

Chat with your PDF documents using Claude AI. Upload any PDF, ask questions in natural language, and get cited answers — powered by your own Anthropic API key.

![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat)
![Claude](https://img.shields.io/badge/Claude-Opus%204.8-purple?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange?style=flat)

## Features

- **Upload PDFs** via drag-and-drop or file picker
- **Ask questions** in natural language across one or multiple documents
- **Cited answers** — every response shows which page and document it came from
- **Conversation memory** — follow-up questions are understood in context
- **Persistent vector store** — uploaded documents survive server restarts
- **Your key, your data** — API key stays in your browser; the server never stores it

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Claude](https://www.anthropic.com) — `claude-opus-4-8` (user-supplied API key) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) — `BAAI/bge-small-en-v1.5` (local, no key needed) |
| Vector DB | [ChromaDB](https://www.trychroma.com) (persisted to disk) |
| RAG Pipeline | [LangChain](https://www.langchain.com) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

## Project Structure

```
RAG/
├── backend/
│   ├── main.py          # FastAPI app — all API routes and RAG pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html       # Single-page app served by FastAPI
├── Procfile             # Railway/Render deployment
├── start.sh             # One-command local startup
└── .gitignore
```

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com)

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/rag-document-chatbot.git
cd rag-document-chatbot

# Start the app (installs dependencies and launches the server)
./start.sh
```

Then open **http://localhost:8000** in your browser.

On first launch the app will prompt you for your Anthropic API key. It is saved to `localStorage` in your browser and sent as an `X-API-Key` header — never stored on the server.

Or run manually:

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

> **First run:** FastEmbed downloads the embedding model (~90 MB) automatically. Subsequent starts are instant.

## How It Works

1. **Upload** — PDFs are split into 1000-token chunks with 150-token overlap and embedded locally using FastEmbed (`BAAI/bge-small-en-v1.5`). Embeddings are stored in ChromaDB.
2. **Query** — When you ask a question, the 4 most semantically similar chunks are retrieved from ChromaDB.
3. **Answer** — The retrieved chunks plus the last 3 turns of conversation history are passed to Claude to generate a grounded answer with source citations.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and index a PDF |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{id}` | Remove a document |
| `POST` | `/chat` | Send a message, get an answer + sources (requires `X-API-Key` header) |
| `DELETE` | `/chat/{session_id}` | Clear conversation history |
| `GET` | `/health` | Server status |

## Deploying to Railway / Render

The `Procfile` is already included. Just push to GitHub and connect the repo:

- **Railway**: New project → Deploy from GitHub repo → Done
- **Render**: New Web Service → Connect repo → Build command: `pip install -r backend/requirements.txt`

No environment variables needed — users supply their own API key through the UI.

## Limitations

- Scanned PDFs (image-based) are not supported — text must be extractable
- Conversation history is in-memory and resets when the server restarts
- Each chat request creates a Claude API call billed to the user's key
