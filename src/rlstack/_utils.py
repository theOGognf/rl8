import time
from contextlib import contextmanager
from typing import Callable, Generator, Iterable

from .specs import CompositeSpec, TensorSpec


def assert_1d_spec(spec: TensorSpec, /) -> None:
    """Check if the spec is valid for default models and distributions
    by asserting that it's 1D.

    Args:
        spec: Observation or action spec when using default models or
            distributions.

    Raises:
        AssertionError: If ``spec`` is not 1D.

    """
    assert spec.ndim == 1, (
        f"{spec} is not compatible with default models and distributions. "
        "Default models and distributions do not support tensor specs "
        "that aren't 1D. Tensor specs must have shape ``[N]`` "
        "(where ``N`` is the number of independent elements) to be "
        "compatible with default models and distributions."
    )


def assert_nd_spec(spec: TensorSpec, /) -> None:
    """Check if the spec is at least 1D so it can properly interface with
    library objects.

    Args:
        spec: Observation or action spec.

    Raises:
        AssertionError: If ``spec`` is not at least 1D.

    """
    match spec:
        case CompositeSpec():
            for k in spec:
                assert_nd_spec(spec[k])
        case _:
            assert spec.ndim >= 1, (
                f"{spec} is not a valid spec. Models and distributions must have specs"
                " that have a non-empty shape. Tensor specs must have shape ``[N,"
                " ...]`` (where ``N`` is the number of independent elements and"
                " ``...`` is any number of additional dimensions)."
            )


@contextmanager
def profile_ms() -> Generator[Callable[[], float], None, None]:
    """Profiling context manager that returns the time it took for the
    code to execute within the context's scope in milliseconds.

    """
    start = time.perf_counter_ns()
    yield lambda: (time.perf_counter_ns() - start) / 1e6


class CumulativeAverage:
    """Helper for maintaining a cumulative average.

    Useful for keeping track of statistics temporarily.

    Examples:
        >>> from rlstack.data import CumulativeAverage
        >>> ca = CumulativeAverage()
        >>> ca.update(0.0)
        0.0
        >>> ca.update(2.0)
        1.0

    """

    #: Current running cumulative average value.
    avg: float

    #: Number of samples.
    n: int

    def __init__(self) -> None:
        self.avg = 0.0
        self.n = 0

    def update(self, value: float, /) -> float:
        self.avg = (value + self.n * self.avg) / (self.n + 1)
        return self.avg


class StatTracker:
    """A utility for tracking running cumulative averages for values.

    Mainly used for tracking losses and coefficients during training.

    Args:
        keys: All keys to keep stat track for.
        sum_keys: Subset of keys from ``keys`` to keep running sums of before
            updating their running cumulative averages. Mostly useful for
            aggregating values across batches that're contributing to the
            same update step (e.g., losses associated with updates via
            accumulated gradients).

    """

    #: Mapping of key to their running cumulative average tracker.
    cumulative_averages: dict[str, CumulativeAverage]

    #: Mapping of key to a temporary sum used for tracking values that
    #: are aggregated in batches before counting towards an average.
    #: These values are most commonly losses that're aggregated over
    #: multiple batches.
    sums: dict[str, float]

    def __init__(
        self, keys: Iterable[str], *, sum_keys: None | Iterable[str] = None
    ) -> None:
        sum_keys = sum_keys or []
        self.cumulative_averages = {k: CumulativeAverage() for k in keys}
        self.sums = {k: 0 for k in sum_keys}

    def items(self) -> dict[str, float]:
        """Return a mapping of keys to their current running cumulative average values.
        """
        return {k: ca.avg for k, ca in self.cumulative_averages.items()}

    def update(self, data: dict[str, float], /, *, reduce: bool = False) -> None:
        """Update running stats.

        Args:
            data: Mapping of keys to their data values.
            reduce: Whether to reduce the sums into the cumulative average
                stat trackers.

        """
        for k in self.sums.keys():
            self.sums[k] += data[k]

        for k in set(self.cumulative_averages.keys()) - set(self.sums.keys()):
            self.cumulative_averages[k].update(data[k])

        if reduce:
            for k in self.sums.keys():
                self.cumulative_averages[k].update(self.sums[k])
                self.sums[k] = 0.0
