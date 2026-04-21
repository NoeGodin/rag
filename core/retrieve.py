from langchain_core.documents import Document

from config import get_vector_store, RETRIEVAL_K_FETCH, RERANK_ENABLED, RETRIEVAL_TYPE
from utils.rerank import rerank
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_vector_store = get_vector_store()


def _build_retriever():
    match RETRIEVAL_TYPE:
        case "similarity":
            return _vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K_FETCH},
            )
        case "mmr":
            return _vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": RETRIEVAL_K_FETCH, "fetch_k": RETRIEVAL_K_FETCH * 2},
            )
        case _:
            raise ValueError(f"Unknown retrieval type: {RETRIEVAL_TYPE}")


_retriever = _build_retriever()


def retrieve(query: str) -> list[Document]:
    with timer(f"retrieve [{RETRIEVAL_TYPE}]"):
        docs = _retriever.invoke(query)
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        logger.debug(f"Document {i+1} | {source} p.{page} | {doc.page_content[:200]}")
    return rerank(query, docs) if RERANK_ENABLED else docs
