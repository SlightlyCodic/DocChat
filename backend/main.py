import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR = "chroma_db"
MODEL = "llama3.2"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

embeddings = OllamaEmbeddings(model=MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
llm = ChatOllama(model=MODEL, temperature=0)

# Per-session conversation history
chat_histories: dict[str, list] = {}

app = FastAPI(title="RAG Document Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


def get_all_documents() -> list[dict]:
    """Derive document list from ChromaDB metadata — survives server restarts."""
    try:
        result = vectorstore._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in (result["metadatas"] or []):
            doc_id = meta.get("doc_id")
            if doc_id:
                if doc_id not in docs:
                    docs[doc_id] = {
                        "id": doc_id,
                        "filename": meta.get("filename", "Unknown"),
                        "chunks": 0,
                    }
                docs[doc_id]["chunks"] += 1
        return list(docs.values())
    except Exception:
        return []


def format_chat_history(history: list) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:  # last 3 turns
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        if not pages:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="PDF has no extractable text. Scanned PDFs are not supported.",
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = file.filename

        vectorstore.add_documents(chunks)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "pages": len(pages),
    }


@app.get("/documents")
async def list_documents():
    return get_all_documents()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        vectorstore._collection.delete(where={"doc_id": doc_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete from store: {e}")

    (UPLOAD_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not get_all_documents():
        raise HTTPException(status_code=400, detail="No documents uploaded. Upload a PDF first.")

    history = chat_histories.setdefault(req.session_id, [])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    relevant_docs = retriever.invoke(req.message)
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    history_str = format_chat_history(history)

    system_content = (
        "You are a helpful document assistant. Answer questions using only the provided context.\n"
        "If the answer is not in the context, say you don't know. Be concise.\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )
    if history_str:
        system_content += f"\n\nCONVERSATION HISTORY:\n{history_str}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({"question": req.message})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    history.extend([HumanMessage(content=req.message), AIMessage(content=answer)])
    chat_histories[req.session_id] = history[-20:]

    sources: list[dict] = []
    seen: set[tuple] = set()
    for doc in relevant_docs:
        key = (doc.metadata.get("filename", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            excerpt = doc.page_content[:250].strip()
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": int(doc.metadata.get("page", 0)) + 1,
                "excerpt": excerpt + ("..." if len(doc.page_content) > 250 else ""),
            })

    return ChatResponse(answer=answer, sources=sources)


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    chat_histories.pop(session_id, None)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}
