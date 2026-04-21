from typing import Generator

from config import get_vector_store, get_llm, get_prompt, RETRIEVAL_K
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_vector_store = get_vector_store()
_llm = get_llm()
_prompt = get_prompt()
_chain = _prompt | _llm


def ask_agent(user_query: str, stream: bool = False) -> str | Generator[str, None, None]:
    logger.info(f"User query: {user_query}")

    with timer("similarity search"):
        retrieved_docs = _vector_store.similarity_search(user_query, k=RETRIEVAL_K)
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        logger.debug(f"Document {i+1} | {source} p.{page} | {doc.page_content[:200]}")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if stream:
        logger.debug("Streaming response...")
        return _stream(context, user_query)

    with timer("llm invoke"):
        response = _chain.invoke({"context": context, "question": user_query})
    logger.debug(f"Answer: {response.content[:200]}")
    return response.content


def _stream(context: str, question: str) -> Generator[str, None, None]:
    for chunk in _chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content
