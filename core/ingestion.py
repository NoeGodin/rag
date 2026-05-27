from config import get_vector_store, get_loader, get_text_splitter
from utils.logger import get_logger
from utils.pdf_to_txt import pdf_to_txt
from utils.timer import timer

logger = get_logger(__name__)


def ingest() -> None:
    with timer("chargement PDF"):
        docs = get_loader().load()
    logger.info(f"Ingestion — {len(docs)} pages chargées")

    for txt_path in pdf_to_txt(docs):
        logger.debug(f"  → {txt_path}")

    with timer("split"):
        all_splits = get_text_splitter().split_documents(docs)
    logger.debug(f"{len(all_splits)} chunks créés")

    with timer("embedding + stockage"):
        vector_store = get_vector_store()
        vector_store.add_documents(documents=all_splits)
    logger.info("Ingestion terminée")


if __name__ == "__main__":
    ingest()
