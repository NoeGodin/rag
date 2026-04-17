from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings

# TODO : would be great to load documents from sharepoint
folder_path = "assets/"
loader = DirectoryLoader(
    folder_path,
    glob="**/*.pdf", # also look at subfolder
    loader_cls=PyPDFLoader
)
docs = loader.load()
print(f"{len(docs)} pages de pdf chargés")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# TODO : Benchmark differents Embedding models
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# TODO : change solution in future
vector_store = Chroma(
    collection_name="coArchi",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
ids = vector_store.add_documents(documents=all_splits)