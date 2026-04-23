from langchain_core.documents import Document

from config import RETRIEVAL_K, RERANK_ENABLED
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from flashrank import Ranker  # noqa: F401
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        FlashrankRerank.model_rebuild()
        _reranker = FlashrankRerank(top_n=RETRIEVAL_K)
    return _reranker


def rerank(query: str, docs: list[Document]) -> list[Document]:
    with timer("rerank"):
        reranked = _get_reranker().compress_documents(docs, query)
    for i, doc in enumerate(reranked):
        score = doc.metadata.get("relevance_score", "?")
        logger.debug(f"Reranked {i+1} | score={score:.4f} | {doc.page_content[:100]}")
    return list(reranked)
