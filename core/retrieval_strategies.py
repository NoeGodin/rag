from enum import Enum


class RetrievalType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
