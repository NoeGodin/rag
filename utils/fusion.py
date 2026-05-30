from langchain_core.documents import Document


def reciprocal_rank_fusion(
    results_list: list[list[Document]], k=60, top_n=5
) -> list[Document]:
    """Applique l'algorithme RRF pour fusionner plusieurs listes de résultats."""
    fused_scores = {}
    doc_map = {}  # Pour garder une référence vers l'objet Document original

    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_key = (doc.metadata.get("source", ""), doc.page_content)

            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0
                doc_map[doc_key] = doc

            fused_scores[doc_key] += 1 / (rank + k)

    # Tri par score RRF décroissant
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # On renvoie les Top N documents
    return [doc_map[doc_key] for doc_key, score in reranked[:top_n]]
