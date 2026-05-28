from langchain_core.documents import Document

from config import get_vector_store, RETRIEVAL_K_FETCH, RERANK_ENABLED, RELEVANCE_THRESHOLD
from utils.rerank import rerank
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_store = get_vector_store()


def retrieve(query: str) -> list[Document]:
    with timer("retrieve [similarity_with_scores]"):
        results = _store.similarity_search_with_relevance_scores(
            query, k=RETRIEVAL_K_FETCH
        )

    docs = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]

    logger.info(
        f"{len(docs)}/{len(results)} document(s) above threshold ({RELEVANCE_THRESHOLD})"
    )
    for i, (doc, score) in enumerate(results[:5]):
        source = doc.metadata.get("source", "inconnu")
        logger.debug(f"  Doc {i+1} | score={score:.3f} | {source}")

    return rerank(query, docs) if RERANK_ENABLED and docs else docs
