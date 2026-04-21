from config import get_vector_store, get_loader, get_text_splitter
from utils.logger import get_logger
from utils.pdf_to_txt import pdf_to_txt

logger = get_logger(__name__)


def ingest() -> None:
    docs = get_loader().load()
    logger.info(f"{len(docs)} pages chargées")

    for txt_path in pdf_to_txt(docs):
        logger.info(f"  → {txt_path}")

    all_splits = get_text_splitter().split_documents(docs)
    logger.info(f"{len(all_splits)} chunks créés")

    vector_store = get_vector_store()
    vector_store.add_documents(documents=all_splits)
    logger.info("Ingestion terminée")


if __name__ == "__main__":
    ingest()
