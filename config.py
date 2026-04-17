from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

# ── Vector store ─────────────────────────────────────────────────────────────
VECTOR_STORE = Chroma(
    collection_name="coArchi",
    embedding_function=EMBEDDING_MODEL,
    persist_directory="./chroma_langchain_db",
)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM = ChatOllama(model="llama3.1")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say you don't know."""

# ── Ingestion settings ────────────────────────────────────────────────────────
DOCUMENTS_FOLDER = "assets/"
DOCUMENTS_GLOB = "**/*.pdf"
DOCUMENT_LOADER_CLS = PyPDFLoader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval settings ────────────────────────────────────────────────────────
RETRIEVAL_K = 2
