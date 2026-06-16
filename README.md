# DocChat — RAG-Powered Document Chatbot

Chat with your PDF documents using a locally hosted LLM. Upload any PDF, ask questions in natural language, and get cited answers — no API keys, no cloud, everything runs on your machine.

![DocChat UI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange?style=flat)

## Features

- **Upload PDFs** via drag-and-drop or file picker
- **Ask questions** in natural language across one or multiple documents
- **Cited answers** — every response shows which page and document it came from
- **Conversation memory** — follow-up questions are understood in context
- **100% local** — LLM and embeddings run via Ollama, no external API calls
- **Persistent vector store** — uploaded documents survive server restarts

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) — `llama3.2` |
| Embeddings | Ollama `llama3.2` (via `/api/embeddings`) |
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
├── start.sh             # One-command startup script
└── .gitignore
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

Pull the model before first run:

```bash
ollama pull llama3.2
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/rag-document-chatbot.git
cd rag-document-chatbot

# Start the app (installs dependencies and launches the server)
./start.sh
```

Then open **http://localhost:8000** in your browser.

Or manually:

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

## How It Works

1. **Upload** — PDFs are split into 1000-token chunks with 150-token overlap and embedded using `llama3.2` via Ollama. Embeddings are stored in ChromaDB.
2. **Query** — When you ask a question, the 4 most semantically similar chunks are retrieved from ChromaDB.
3. **Answer** — The retrieved chunks plus the last 3 turns of conversation history are passed to `llama3.2` to generate a grounded answer with source citations.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and index a PDF |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{id}` | Remove a document |
| `POST` | `/chat` | Send a message, get an answer + sources |
| `DELETE` | `/chat/{session_id}` | Clear conversation history |
| `GET` | `/health` | Server and model status |

## Limitations

- Scanned PDFs (image-based) are not supported — text must be extractable
- First response after a cold start may be slow while Ollama loads the model into memory
- Conversation history is in-memory and resets when the server restarts
