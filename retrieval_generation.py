import logging

from config import VECTOR_STORE, LLM, SYSTEM_PROMPT, RETRIEVAL_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ask_agent(user_query: str, stream: bool = False):
    logger.info(f"user query: {user_query}")
    retrieved_docs = VECTOR_STORE.similarity_search(user_query, k=RETRIEVAL_K)
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"\n--- Document {i+1} ---")
        logger.info(f"Metadata: {doc.metadata}")
        logger.info(f"Content: {doc.page_content[:500]}")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = SYSTEM_PROMPT.format(context=context, question=user_query)

    if stream:
        logger.info("Streaming response...")
        for chunk in LLM.stream(prompt):
            if chunk.content:
                yield chunk.content
    else:
        logger.info("Generating full response...")
        response = LLM.invoke(prompt)
        logger.info(f"Final answer: {response.content}")
        return response.content
