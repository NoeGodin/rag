import time
from contextlib import contextmanager
from typing import Generator

from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.debug(f"[timer] {label} — {elapsed:.3f}s")
