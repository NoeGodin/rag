from langchain_core.documents import Document

from config import get_vector_store, RETRIEVAL_TYPE, RETRIEVAL_K_FETCH, RERANK_ENABLED
from core.retrieval_strategies import build_retriever
from utils.rerank import rerank
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_store = get_vector_store()


def retrieve(query: str) -> list[Document]:
    retriever = build_retriever(_store, RETRIEVAL_TYPE, RETRIEVAL_K_FETCH)

    with timer(f"retrieve [{RETRIEVAL_TYPE}]"):
        docs = retriever.invoke(query)

    logger.info(f"{len(docs)} document(s) récupéré(s)")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        logger.debug(f"Document {i+1} | {source} p.{page} | {doc.page_content[:200]}")

    return rerank(query, docs) if RERANK_ENABLED else docs
