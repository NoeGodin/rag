from typing import Generator

from config import get_gemini_llm, get_prompt
from utils.logger import get_logger
from utils.timer import timer

logger = get_logger(__name__)

_chain = get_prompt() | get_gemini_llm()


def generate(context: str, question: str) -> str:
    with timer("llm invoke"):
        response = _chain.invoke({"context": context, "question": question})
    logger.debug(f"Answer: {response.content[:200]}")
    return response.content


def generate_stream(context: str, question: str) -> Generator[str, None, None]:
    logger.debug("Streaming response...")
    for chunk in _chain.stream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content
