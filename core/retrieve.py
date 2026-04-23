from langchain_core.documents import Document

from config import get_vector_store, RETRIEVAL_TYPE, RETRIEVAL_K_FETCH, RERANK_ENABLED
from core.retrieval_strategies import build_retriever
from utils.project_filter import detect_project_in_query
from utils.rerank import rerank
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_store = get_vector_store()


def _get_known_stems() -> list[str]:
    try:
        result = _store.get(include=["metadatas"])
        stems = {
            m.get("source_stem", "")
            for m in result.get("metadatas", [])
            if m.get("source_stem")
        }
        return list(stems)
    except Exception:
        return []


_known_stems: list[str] = _get_known_stems()


def retrieve(query: str) -> list[Document]:
    matched_token = detect_project_in_query(query, _known_stems)

    search_kwargs: dict = {"k": RETRIEVAL_K_FETCH}
    if matched_token:
        matching_stems = [s for s in _known_stems if matched_token.lower() in s.lower()]
        if matching_stems:
            search_kwargs["filter"] = {"source_stem": {"$in": matching_stems}}
            logger.debug(f"Filtre projet appliqué : {matching_stems}")
    else:
        logger.debug("Aucun token projet détecté dans la query")

    retriever = build_retriever(_store, RETRIEVAL_TYPE, RETRIEVAL_K_FETCH, extra_search_kwargs=search_kwargs)

    with timer(f"retrieve [{RETRIEVAL_TYPE}]"):
        docs = retriever.invoke(query)

    logger.info(f"{len(docs)} document(s) récupéré(s)")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        logger.debug(f"Document {i+1} | {source} p.{page} | {doc.page_content[:200]}")

    return rerank(query, docs) if RERANK_ENABLED else docs
