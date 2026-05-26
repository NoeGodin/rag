from google import genai
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from core.retrieval_strategies import RetrievalType

import os

load_dotenv()

RETRIEVAL_K = 2
RETRIEVAL_K_FETCH = 10
RERANK_ENABLED = True
RETRIEVAL_TYPE = RetrievalType.SIMILARITY


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


# def get_embeddings() -> OllamaEmbeddings:
#     return OllamaEmbeddings(model="nomic-embed-text")

def get_gemini_embeddings():
    API_KEY = gemini_api_key()
    return GoogleGenerativeAIEmbeddings(api_key=API_KEY, model="models/gemini-embedding-001")

def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="arb",
        embedding_function=get_gemini_embeddings(),
        persist_directory="./chroma_langchain_db",
    )


# def get_llm() -> ChatOllama:
#     return ChatOllama(model="llama3.1")

def get_gemini_llm():
    API_KEY = gemini_api_key()
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        api_key=API_KEY
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
    
def gemini_api_key() -> str:
    return os.getenv("API_KEY")