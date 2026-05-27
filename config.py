from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from dotenv import load_dotenv

from core.retrieval_strategies import RetrievalType

import os

load_dotenv()

RETRIEVAL_K = 2
RETRIEVAL_K_FETCH = 10
RERANK_ENABLED = True
RETRIEVAL_TYPE = RetrievalType.SIMILARITY

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="arb",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_langchain_db",
    )


def get_llm() -> ChatOpenAI:
    fal_key = openrouter_api_key()
    return ChatOpenAI(
        model="meta-llama/llama-4-maverick",
        base_url="https://fal.run/openrouter/router/openai/v1",
        api_key="fal",  # placeholder, auth via header
        temperature=0.1,
        streaming=True,
        default_headers={"Authorization": f"Key {fal_key}"},
    )
    

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

def openrouter_api_key() -> str:
    return os.getenv("FAL_KEY")