from langchain_core.documents import Document

from config import get_vector_store, RETRIEVAL_K
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_vector_store = get_vector_store()


def retrieve(query: str) -> list[Document]:
    with timer("similarity search"):
        docs = _vector_store.similarity_search(query, k=RETRIEVAL_K)
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        logger.debug(f"Document {i+1} | {source} p.{page} | {doc.page_content[:200]}")
    return docs
