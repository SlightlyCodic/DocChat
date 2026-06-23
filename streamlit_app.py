import os
import uuid
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="DocChat", page_icon="📚", layout="wide")

CLAUDE_MODEL = "claude-opus-4-8"


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def init_state():
    defaults = {
        "chat_history": [],
        "documents": {},       # doc_id -> filename
        "indexed_files": set(),
        "vectorstore": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_vectorstore():
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma(embedding_function=load_embeddings())
    return st.session_state.vectorstore


init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 DocChat")
    st.caption("Chat with your PDFs using Claude AI")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-api03-…",
        help="Used only for this session. Never stored on the server.",
    )

    if not api_key:
        st.info("Enter your Anthropic API key above to get started.")
        st.stop()

    st.divider()
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    for f in uploaded_files or []:
        if f.name not in st.session_state.indexed_files:
            with st.spinner(f"Indexing {f.name}…"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                try:
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    if not pages:
                        st.error(f"{f.name}: no extractable text (scanned PDF?)")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=150
                        )
                        chunks = splitter.split_documents(pages)
                        doc_id = str(uuid.uuid4())
                        for chunk in chunks:
                            chunk.metadata["doc_id"] = doc_id
                            chunk.metadata["filename"] = f.name
                        get_vectorstore().add_documents(chunks)
                        st.session_state.documents[doc_id] = f.name
                        st.session_state.indexed_files.add(f.name)
                        st.success(f"✓ {f.name} — {len(pages)} pages, {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.documents:
        st.caption("Indexed:")
        for doc_id, filename in list(st.session_state.documents.items()):
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📄 {filename}")
            if col2.button("×", key=f"del_{doc_id}", help="Remove"):
                get_vectorstore()._collection.delete(where={"doc_id": doc_id})
                del st.session_state.documents[doc_id]
                st.session_state.indexed_files.discard(filename)
                st.rerun()

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Chat with your documents")

if not st.session_state.documents:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if question := st.chat_input("Ask something about your documents…"):
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            vs = get_vectorstore()
            relevant_docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(question)
            context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

            history_msgs = st.session_state.chat_history[:-1][-6:]
            history_str = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in history_msgs
            )

            system_content = (
                "You are a helpful document assistant. Answer questions using only the provided context.\n"
                "If the answer is not in the context, say you don't know. Be concise.\n\n"
                f"DOCUMENT CONTEXT:\n{context}"
            )
            if history_str:
                system_content += f"\n\nCONVERSATION HISTORY:\n{history_str}"

            llm = ChatAnthropic(model=CLAUDE_MODEL, api_key=api_key, temperature=0)
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", system_content),
                    ("human", "{question}"),
                ])
                | llm
                | StrOutputParser()
            )

            try:
                answer = chain.invoke({"question": question})
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.markdown(answer)

        if relevant_docs:
            seen: set[tuple] = set()
            sources = []
            for doc in relevant_docs:
                key = (doc.metadata.get("filename", ""), doc.metadata.get("page", 0))
                if key not in seen:
                    seen.add(key)
                    sources.append(doc)
            with st.expander(f"📚 {len(sources)} source{'s' if len(sources) != 1 else ''}"):
                for doc in sources:
                    st.markdown(
                        f"**{doc.metadata.get('filename', 'Unknown')}**"
                        f" — p.{int(doc.metadata.get('page', 0)) + 1}"
                    )
                    st.caption(f"_{doc.page_content[:200].strip()}…_")

        st.session_state.chat_history.append(AIMessage(content=answer))
