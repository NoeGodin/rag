from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

from config import VECTOR_STORE, DOCUMENTS_FOLDER, DOCUMENTS_GLOB, DOCUMENT_LOADER_CLS, CHUNK_SIZE, CHUNK_OVERLAP

# TODO: load documents from SharePoint
loader = DirectoryLoader(
    DOCUMENTS_FOLDER,
    glob=DOCUMENTS_GLOB,
    loader_cls=DOCUMENT_LOADER_CLS,
)
docs = loader.load()
print(f"{len(docs)} pages de pdf chargés")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

ids = VECTOR_STORE.add_documents(documents=all_splits)
