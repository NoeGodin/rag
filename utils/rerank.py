from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import RETRIEVAL_K
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)


def rerank(query: str, docs: list[Document]) -> list[Document]:
    with timer("rerank"):
        tokenized_docs = [doc.page_content.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query.lower().split())

    for i, (doc, score) in enumerate(zip(docs, scores)):
        logger.debug(f"BM25 score {i+1} | score={score:.4f} | {doc.page_content[:100]}")

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:RETRIEVAL_K]]
