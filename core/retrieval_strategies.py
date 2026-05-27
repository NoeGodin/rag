from enum import Enum

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


class RetrievalType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"


def build_retriever(
    store: Chroma,
    retrieval_type: RetrievalType,
    k_fetch: int,
) -> BaseRetriever:
    match retrieval_type:
        case RetrievalType.SIMILARITY:
            return store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_fetch},
            )
        case RetrievalType.MMR:
            return store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_fetch, "fetch_k": k_fetch * 2},
            )
        case _:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")
