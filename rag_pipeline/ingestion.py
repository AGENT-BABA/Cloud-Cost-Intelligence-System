"""
ingestion.py
------------
Loads the three context documents (general_context.txt, actions.txt,
business_context_template.txt), splits them into chunks, embeds them with
a local nomic-embed-text model via Ollama, and persists two separate
FAISS vector stores:

  vector-store/vs_technical/   — general_context + actions  (for Verdict 1)
  vector-store/vs_business/    — business_context + actions  (for Verdict 2)

Run this script once (or whenever the context files change) before running
verdict1.py or verdict2.py.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXT_DIR = os.path.join(BASE_DIR, "context")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector-store")

GENERAL_CONTEXT_PATH = os.path.join(CONTEXT_DIR, "general_context.txt")
ACTIONS_PATH = os.path.join(CONTEXT_DIR, "actions.txt")
BUSINESS_CONTEXT_PATH = os.path.join(CONTEXT_DIR, "business_context_template.txt")

VS_TECHNICAL_PATH = os.path.join(VECTOR_STORE_DIR, "vs_technical")
VS_BUSINESS_PATH = os.path.join(VECTOR_STORE_DIR, "vs_business")

# Embedding model — nomic-embed-text runs locally via Ollama
EMBED_MODEL = "nomic-embed-text"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_document(path: str) -> list:
    """Load a single text file and return a list of LangChain Documents."""
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 80) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


def build_vector_store(docs: list, embeddings) -> FAISS:
    """Build a FAISS vector store from a list of Document chunks."""
    return FAISS.from_documents(docs, embeddings)


# ── Main ingestion logic ──────────────────────────────────────────────────────

def ingest():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    print("[ingestion] Initialising Ollama embeddings model:", EMBED_MODEL)
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
    )

    # --- Load raw documents ---
    print("[ingestion] Loading context documents …")
    general_docs = load_document(GENERAL_CONTEXT_PATH)
    actions_docs = load_document(ACTIONS_PATH)
    business_docs = load_document(BUSINESS_CONTEXT_PATH)

    # --- Chunk ---
    print("[ingestion] Splitting documents …")
    general_chunks = split_documents(general_docs)
    actions_chunks = split_documents(actions_docs)
    business_chunks = split_documents(business_docs)

    # --- Build Technical vector store (Verdict 1) ---
    print("[ingestion] Building technical vector store (general_context + actions) …")
    technical_chunks = general_chunks + actions_chunks
    vs_technical = build_vector_store(technical_chunks, embeddings)
    vs_technical.save_local(VS_TECHNICAL_PATH)
    print(f"[ingestion] Technical store saved → {VS_TECHNICAL_PATH}")

    # --- Build Business vector store (Verdict 2) ---
    print("[ingestion] Building business vector store (business_context + general_context + actions) …")
    business_all_chunks = business_chunks + general_chunks + actions_chunks
    vs_business = build_vector_store(business_all_chunks, embeddings)
    vs_business.save_local(VS_BUSINESS_PATH)
    print(f"[ingestion] Business store saved → {VS_BUSINESS_PATH}")

    print("[ingestion] ✓ Ingestion complete.")


if __name__ == "__main__":
    ingest()
