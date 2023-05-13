import time
from contextlib import contextmanager
from typing import Callable, Generator

from .specs import TensorSpec


def assert_1d_spec(spec: TensorSpec, /) -> None:
    """Check if the spec is valid for default models and distributions
    by asserting that it's 1D.

    Args:
        spec: Observation or action spec when using default models or
            distributions.

    Raises:
        AssertionError: If ``spec`` is not 1D.

    """
    assert spec.shape.numel() == 1, (
        "Default models and distributions do not support tensor specs "
        "that aren't 1D. Tensor specs must have shape ``[N]`` "
        "(where ``N`` is the number of independent elements) to be "
        "compatible with default models and distributions."
    )


@contextmanager
def profile_ms() -> Generator[Callable[[], float], None, None]:
    """Profiling context manager that returns the time it took for the
    code to execute within the context's scope in milliseconds.

    """
    start = time.perf_counter_ns()
    yield lambda: (time.perf_counter_ns() - start) / 1e6
