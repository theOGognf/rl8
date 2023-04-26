"""General utils."""

import time
from contextlib import contextmanager
from typing import Callable, Generator


@contextmanager
def profile_ms() -> Generator[Callable[[], float], None, None]:
    """Profiling context manager in milliseconds."""
    start = time.perf_counter_ns()
    yield lambda: (time.perf_counter_ns() - start) / 1e6
