"""
PDF Question-Answering (RAG)
============================
Upload a PDF and ask questions about its content.
Uses Retrieval-Augmented Generation: the app splits the PDF into chunks,
embeds them into a vector store, retrieves relevant chunks for each question,
and feeds them to the LLM as context.

Key LangChain concepts:
  - PyPDFLoader              (document loading)
  - RecursiveCharacterTextSplitter (chunking)
  - GoogleGenerativeAIEmbeddings   (embeddings)
  - FAISS                    (vector store)
  - create_stuff_documents_chain + create_retrieval_chain (RAG pipeline)

Run:  streamlit run pdf_qa/app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ── Load environment variables (.env should contain GOOGLE_API_KEY) ──────────
load_dotenv(encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def load_and_split_pdf(uploaded_file) -> list:
    """
    Save the uploaded file to a temp path, load it with PyPDFLoader,
    then split into chunks with RecursiveCharacterTextSplitter.
    """
    # PyPDFLoader needs a file path, so write the upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # 1️⃣  Load — turns the PDF into a list of Document objects (one per page)
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # 2️⃣  Split — break pages into smaller, overlapping chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(pages)
        return chunks
    finally:
        os.unlink(tmp_path)


def build_vector_store(chunks):
    """
    Embed all chunks and store them in a FAISS vector store.
    Returns the vector store instance.
    """
    # 3️⃣  Embed — convert text chunks into numerical vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4️⃣  Store — index the vectors for fast similarity search
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_rag_chain(vector_store):
    """
    Build a retrieval-augmented generation chain:
      retriever ➜ stuff documents into prompt ➜ LLM ➜ answer
    """
    # 5️⃣  LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    # 6️⃣  Prompt — instruct the LLM to answer based on provided context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions based on the "
         "provided document context. If the answer is not in the context, "
         "say you don't know. Always cite which part of the document your "
         "answer comes from when possible."),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {input}"),
    ])

    # 7️⃣  Stuff documents chain — takes retrieved docs, stuffs them into the prompt
    stuff_chain = create_stuff_documents_chain(llm, prompt)

    # 8️⃣  Retrieval chain — retriever fetches docs, then passes to stuff chain
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    rag_chain = create_retrieval_chain(retriever, stuff_chain)

    return rag_chain


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄", layout="centered")
st.title("📄 PDF Question-Answering")
st.caption("Upload a PDF, ask questions — powered by RAG + Google Gemini")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("🗑️ Clear everything"):
        for key in ["vector_store", "chunks", "messages", "pdf_name"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Your PDF is split into overlapping chunks "
        "(**RecursiveCharacterTextSplitter**).\n"
        "2. Each chunk is turned into a vector "
        "(**GoogleGenerativeAIEmbeddings**).\n"
        "3. Vectors are stored in a **FAISS** index.\n"
        "4. When you ask a question, the 4 most similar chunks are "
        "retrieved and passed to **Gemini** as context.\n"
        "5. The LLM answers based *only* on the retrieved context."
    )

# ── Initialise session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Process uploaded PDF ─────────────────────────────────────────────────────
if uploaded_file is not None:
    # Only re-process if it's a new file
    if st.session_state.get("pdf_name") != uploaded_file.name:
        with st.spinner("Reading and indexing PDF…"):
            chunks = load_and_split_pdf(uploaded_file)
            vector_store = build_vector_store(chunks)

            st.session_state["chunks"] = chunks
            st.session_state["vector_store"] = vector_store
            st.session_state["pdf_name"] = uploaded_file.name
            st.session_state["messages"] = []

        st.success(
            f"Indexed **{uploaded_file.name}** — "
            f"{len(chunks)} chunks from {len(set(c.metadata.get('page', 0) for c in chunks))} pages. "
            f"Ask away!"
        )

# ── Display chat history ────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📑 Source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** (page {page}):")
                    st.text(src.page_content[:500])
                    st.divider()

# ── Handle user question ────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about the PDF…"):
    if "vector_store" not in st.session_state:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build RAG chain and get answer
    rag_chain = build_rag_chain(st.session_state["vector_store"])

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer…"):
            result = rag_chain.invoke({"input": user_input})

        answer = result["answer"]
        source_docs = result.get("context", [])

        st.markdown(answer)

        # Show the source chunks used
        if source_docs:
            with st.expander("📑 Source chunks used"):
                for i, doc in enumerate(source_docs, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** (page {page}):")
                    st.text(doc.page_content[:500])
                    st.divider()

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": source_docs,
    })
