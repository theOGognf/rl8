import time
from contextlib import contextmanager
from typing import Callable, Generator, Iterable, Literal

import psutil
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, TensorSpec
from typing_extensions import Self

from .data import MemoryStats


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


def memory_stats(device_type: Literal["cuda", "cpu"], /) -> MemoryStats:
    """Return memory stats for a particular device type."""
    match device_type:
        case "cpu":
            svmem = psutil.virtual_memory()
            free = svmem.free
            total = svmem.total
        case "cuda":
            free, total = torch.cuda.mem_get_info()
    return {
        "memory/free": free,
        "memory/total": total,
        "memory/percent": 100 * (total - free) / total,
    }


@contextmanager
def profile_ms() -> Generator[Callable[[], float], None, None]:
    """Profiling context manager that returns the time it took for the
    code to execute within the context's scope in milliseconds.

    """
    start = time.perf_counter_ns()
    yield lambda: (time.perf_counter_ns() - start) / 1e6


class Batcher:
    """A simple utility for batching a tensordict.

    Args:
        batch: Tensordict to batch.
        batch_size: Batch size per iteration. Defaults to the whole
            tensordict.
        shuffle: Whether to shuffle samples before batching.

    """

    #: Tensordict to batch when iterating over the batcher.
    batch: TensorDict

    #: Chunk size to split :attr:`Batcher.batch` into.
    batch_size: int

    #: List of indices for each batch. Instantiated each time the batcher
    #: is iterated over.
    indices: list[torch.Tensor]

    #: Whether to shuffle samples before batching for each iteration.
    shuffle: bool

    def __init__(
        self,
        batch: TensorDict,
        /,
        *,
        batch_size: None | int = None,
        shuffle: bool = False,
    ) -> None:
        self.batch = batch
        self.batch_size = batch_size or self.batch.size(0)
        self.shuffle = shuffle

    def __iter__(self) -> Self:
        self.idx = 0
        if self.shuffle:
            indices = torch.randperm(self.batch.size(0), device=self.batch.device)
        else:
            indices = torch.arange(self.batch.size(0), device=self.batch.device)
        self.indices = torch.split(indices, self.batch_size)
        return self

    def __next__(self) -> TensorDict:
        if self.idx < len(self.indices):
            out = self.batch[self.indices[self.idx], ...]
            self.idx += 1
            return out
        raise StopIteration


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
        self.n += 1
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
