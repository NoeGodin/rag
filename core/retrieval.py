from typing import Generator

from config import get_vector_store, get_llm, SYSTEM_PROMPT, RETRIEVAL_K
from utils.logger import get_logger

logger = get_logger(__name__)

_vector_store = get_vector_store()
_llm = get_llm()


def ask_agent(user_query: str, stream: bool = False) -> str | Generator[str, None, None]:
    logger.info(f"User query: {user_query}")

    retrieved_docs = _vector_store.similarity_search(user_query, k=RETRIEVAL_K)
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Document {i+1} | {doc.metadata} | {doc.page_content[:200]}")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = SYSTEM_PROMPT.format(context=context, question=user_query)

    if stream:
        logger.info("Streaming response...")
        return _stream(prompt)

    logger.info("Generating response...")
    response = _llm.invoke(prompt)
    logger.info(f"Answer: {response.content[:200]}")
    return response.content


def _stream(prompt: str) -> Generator[str, None, None]:
    for chunk in _llm.stream(prompt):
        if chunk.content:
            yield chunk.content
