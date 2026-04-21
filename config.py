from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

RETRIEVAL_K = 2
RETRIEVAL_K_FETCH = 10
RERANK_ENABLED = True
RETRIEVAL_TYPE = "similarity"  # "similarity" | "mmr"


def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant.\n\n"
            "Answer the question using ONLY the context below.\n\n"
            "Context:\n{context}\n\n"
            "If the answer is not in the context, say you don't know.",
        ),
        ("human", "{question}"),
    ])


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


def get_loader() -> DirectoryLoader:
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