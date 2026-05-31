from enum import Enum


class RetrievalType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    RAG_FUSION = "rag_fusion"
    HYDE = "hyde"
