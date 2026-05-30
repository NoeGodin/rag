from langchain_core.documents import Document

from config import get_vector_store, RETRIEVAL_K_FETCH, RERANK_ENABLED, RELEVANCE_THRESHOLD, RETRIEVAL_TYPE
from core.retrieval_strategies import RetrievalType
from utils.rerank import rerank
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_store = get_vector_store()


def retrieve(query: str) -> list[Document]:
    match RETRIEVAL_TYPE:
        case RetrievalType.SIMILARITY:
            with timer("retrieve [similarity]"):
                results = _store.similarity_search_with_relevance_scores(
                    query, k=RETRIEVAL_K_FETCH
                )
                # for doc, score in results:
                #     print(f"[SCORE={score:.3f}] {doc.page_content[:80]}...")

            docs = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]

        case RetrievalType.MMR:
            with timer("retrieve [mmr]"):
                retriever = _store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": RETRIEVAL_K_FETCH, "fetch_k": RETRIEVAL_K_FETCH * 2},
                )
                docs = retriever.invoke(query)

    logger.info(f"{len(docs)} document(s) récupérés (mode: {RETRIEVAL_TYPE.value})")
    for i, doc in enumerate(docs[:5]):
        source = doc.metadata.get("source", "inconnu")
        logger.debug(f"  Doc {i+1} | {source}")

    return rerank(query, docs) if RERANK_ENABLED and docs else docs
