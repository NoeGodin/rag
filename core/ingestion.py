import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

from config import VECTOR_STORE, DOCUMENTS_FOLDER, DOCUMENTS_GLOB, DOCUMENT_LOADER_CLS, CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger(__name__)


def make_chunk_id(source: str, index: int) -> str:
    return hashlib.md5(f"{source}:{index}".encode()).hexdigest()


def ingest() -> None:
    # TODO: load documents from SharePoint
    loader = DirectoryLoader(
        DOCUMENTS_FOLDER,
        glob=DOCUMENTS_GLOB,
        loader_cls=DOCUMENT_LOADER_CLS,
    )
    docs = loader.load()
    logger.info(f"{len(docs)} pages de pdf chargées")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"{len(all_splits)} chunks créés")

    ids = [make_chunk_id(doc.metadata["source"], i) for i, doc in enumerate(all_splits)]
    VECTOR_STORE.add_documents(documents=all_splits, ids=ids)
    logger.info(f"Ingestion terminée — {len(ids)} chunks indexés")


if __name__ == "__main__":
    ingest()
