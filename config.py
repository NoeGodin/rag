import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from core.retrieval_strategies import RetrievalType

load_dotenv()

RETRIEVAL_K = 5
RETRIEVAL_K_FETCH = 20
RERANK_ENABLED = True
RETRIEVAL_TYPE = RetrievalType.SIMILARITY

FAL_KEY = os.environ["FAL_KEY"]
FAL_BASE_URL = "https://fal.run/openrouter/router/openai/v1"
FAL_HEADERS = {"Authorization": f"Key {FAL_KEY}"}


def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es un assistant spécialisé en histoire politique et régimes autoritaires.\n\n"
            "Réponds à la question en te basant UNIQUEMENT sur le contexte ci-dessous.\n"
            "Cite tes sources quand c'est possible.\n\n"
            "Contexte :\n{context}\n\n"
            "Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.",
        ),
        ("human", "{question}"),
    ])


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url=FAL_BASE_URL,
        api_key="fal",
        default_headers=FAL_HEADERS,
    )

def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="dictateurs",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_langchain_db",
    )


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="meta-llama/llama-4-maverick",
        base_url=FAL_BASE_URL,
        api_key="fal",
        temperature=0.1,
        streaming=True,
        default_headers=FAL_HEADERS,
    )

def get_loader() -> DirectoryLoader:
    return DirectoryLoader(
        "assets/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )


def get_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
