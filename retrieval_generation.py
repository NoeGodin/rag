from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO : Benchmark differents Embedding models
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# TODO : change solution in future
vector_store = Chroma(collection_name="coArchi",embedding_function=embeddings,persist_directory="./chroma_langchain_db")
# TODO : evaluate model
model = ChatOllama(model="llama3.1")

def ask_agent(user_query: str, stream:bool=False):
    logger.info(f"user query: {user_query}")
    retrieved_docs = vector_store.similarity_search(user_query, k=2)
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"\n--- Document {i+1} ---")
        logger.info(f"Metadata: {doc.metadata}")
        logger.info(f"Content: {doc.page_content[:500]}")  # limite pour lisibilité
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{user_query}

If the answer is not in the context, say you don't know.
"""
    if stream:
        logger.info("Streaming response...")
        for chunk in model.stream(prompt):
            if chunk.content:
                yield chunk.content
    else:
        logger.info("Generating full response...")
        response = model.invoke(prompt)

        logger.info("Response generated")
        logger.info(f"Final answer: {response.content}")

        return response.content

