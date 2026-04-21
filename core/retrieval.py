from typing import Generator

from .retrieve import retrieve
from .generate import generate, generate_stream
from utils.logger import get_logger

logger = get_logger(__name__)


def ask_agent(user_query: str, stream: bool = False) -> str | Generator[str, None, None]:
    logger.info(f"User query: {user_query}")

    docs = retrieve(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    if stream:
        return generate_stream(context, user_query)

    return generate(context, user_query)
