import hashlib
import time
import uuid

from langchain_core.documents import Document

from config import get_vector_store, get_loader, get_text_splitter, get_qdrant_client, QDRANT_COLLECTION
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

BATCH_SIZE = 50
MAX_RETRIES = 3


def _deterministic_id(doc: Document) -> str:
    """SHA-256 hash de page_content + source → ID stable pour upsert."""
    content = doc.page_content + doc.metadata.get("source", "")
    hash_bytes = hashlib.sha256(content.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


def _get_existing_ids(client, ids: list[str]) -> set[str]:
    """Vérifie quels IDs existent déjà dans Qdrant (sans embedder)."""
    existing: set[str] = set()
    # Qdrant scroll avec filtre par IDs, par batch
    for i in range(0, len(ids), 100):
        batch_ids = ids[i:i + 100]
        results = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing.update(str(p.id) for p in results)
    return existing


def ingest() -> None:
    with timer("chargement documents"):
        docs = get_loader().load()
    logger.info(f"Ingestion — {len(docs)} pages chargées")

    with timer("split"):
        all_splits = get_text_splitter().split_documents(docs)
    logger.info(f"{len(all_splits)} chunks créés")

    ids = [_deterministic_id(doc) for doc in all_splits]

    # Filtrer les chunks déjà dans Qdrant → pas d'embedding inutile
    with timer("vérification IDs existants"):
        client = get_qdrant_client()
        if client.collection_exists(QDRANT_COLLECTION):
            existing_ids = _get_existing_ids(client, ids)
        else:
            existing_ids = set()

    new_docs = []
    new_ids = []
    for doc, doc_id in zip(all_splits, ids):
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    logger.info(f"{len(existing_ids)} chunks déjà dans Qdrant — {len(new_docs)} nouveaux à ingérer")

    if not new_docs:
        logger.info("Rien à ingérer, tout est à jour")
        return

    total_batches = (len(new_docs) - 1) // BATCH_SIZE + 1

    with timer("embedding + stockage"):
        vector_store = get_vector_store()
        for i in range(0, len(new_docs), BATCH_SIZE):
            batch_docs = new_docs[i:i + BATCH_SIZE]
            batch_ids = new_ids[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                    logger.info(f"Batch {batch_num}/{total_batches} ingéré ({len(batch_docs)} chunks)")
                    break
                except Exception as e:
                    logger.warning(f"Batch {batch_num} — tentative {attempt}/{MAX_RETRIES} échouée: {e}")
                    if attempt == MAX_RETRIES:
                        logger.error(f"Batch {batch_num} — abandon après {MAX_RETRIES} tentatives")
                    else:
                        time.sleep(5 * attempt)

    logger.info("Ingestion terminée")


if __name__ == "__main__":
    ingest()
