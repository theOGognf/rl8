"""Timing/profiling/benchmarking utils."""

from contextlib import contextmanager
from typing import Generator, Callable
import time

@contextmanager
def profile_ms() -> Generator[Callable[[], float], None, None]:
    """Profiling context manager in milliseconds."""
    start = time.perf_counter_ns()
    yield lambda: 1e6 * (time.perf_counter_ns() - start)
