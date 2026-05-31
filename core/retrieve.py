import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    get_vector_store,
    get_llm,
    RETRIEVAL_K_FETCH,
    RERANK_ENABLED,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_TYPE,
)
from core.retrieval_strategies import RetrievalType
from utils.rerank import rerank
from utils.logger import get_logger
from utils.timer import timer
from utils.fusion import reciprocal_rank_fusion

logger = get_logger(__name__)

_store = get_vector_store()


def generate_queries(original_query: str) -> list[str]:
    """Génère des reformulations de la requête originale pour le RAG-Fusion"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un assistant IA. Ta mission est de générer 3 questions alternatives basées sur la question utilisateur initiale, pour améliorer la recherche d'information dans une base vectorielle.\n"
                "Fournis UNIQUEMENT les requêtes séparées par des retours à la ligne, SANS numérotation (pas de '1.', '2.', etc.), SANS tirets, SANS préambule et SANS guillemets.",
            ),
            ("user", "Question originale : {query}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": original_query})

    # On split la réponse en lignes et on nettoie les préfixes (tirets, nombres) et les espaces
    queries = []
    for q in response.split("\n"):
        clean_q = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", q.strip())
        # Ignorer les lignes trop courtes ou qui ressemblent à un préambule ("Voici 3 questions...")
        if clean_q and not clean_q.lower().startswith("voici"):
            queries.append(clean_q)

    queries.insert(0, original_query)  # On inclut la requête originale
    return queries


def retrieve(
    query: str, retrieval_type: RetrievalType = RETRIEVAL_TYPE
) -> list[Document]:
    match retrieval_type:
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
                    search_kwargs={
                        "k": RETRIEVAL_K_FETCH,
                        "fetch_k": RETRIEVAL_K_FETCH * 2,
                    },
                )
                docs = retriever.invoke(query)

        case RetrievalType.RAG_FUSION:
            with timer("retrieve [RAG-Fusion]"):
                queries = generate_queries(query)
                # log les requêtes générées
                logger.debug("Requêtes générées pour RAG-Fusion :")
                for i, q in enumerate(queries):
                    logger.debug(f"  {i + 1}. {q}")

                retriever = _store.as_retriever(
                    search_type="similarity", search_kwargs={"k": RETRIEVAL_K_FETCH}
                )
                all_results = retriever.batch(queries)
                docs = reciprocal_rank_fusion(all_results, top_n=RETRIEVAL_K_FETCH)

    logger.info(f"{len(docs)} document(s) récupérés (mode: {retrieval_type.value})")
    for i, doc in enumerate(docs[:5]):
        source = doc.metadata.get("source", "inconnu")
        logger.debug(f"  Doc {i + 1} | {source}")

    return rerank(query, docs) if RERANK_ENABLED and docs else docs
