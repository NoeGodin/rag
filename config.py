from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

SYSTEM_PROMPT = """You are an AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say you don't know."""

RETRIEVAL_K = 2


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="coArchi",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_langchain_db",
    )


def get_llm() -> ChatOllama:
    return ChatOllama(model="llama3.1")


def get_loader() -> BaseLoader:
    return DirectoryLoader(
        "assets/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )


def get_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )