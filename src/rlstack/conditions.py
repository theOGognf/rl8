"""Definitions for monitoring training metrics and determining whether metrics
achieve some condition (most commonly useful for determining when to stop
training).

"""

from typing import Protocol

from .data import TrainStatKey, TrainStats


class Condition(Protocol):
    """Condition callable that's called once per :meth:`Trainer.run`
    iteration. If the condition returns ``True``, then the training
    loop within :meth:`Trainer.run` is stopped.

    """

    def __call__(self, train_stats: TrainStats, /) -> bool:
        """Method to implement that should return ``True`` for forcing training
        within :meth:`Trainer.run` to stop.

        """


class And(Condition):
    """Convenience for joining results from multiple conditions with an ``AND``.

    Args:
        conditions: Conditions to join results for with an ``AND``.

    """

    #: Conditions to join results for with an ``AND``.
    conditions: list[Condition]

    def __init__(self, conditions: list[Condition], /) -> None:
        self.conditions = conditions

    def __call__(self, train_stats: TrainStats, /) -> bool:
        return all([condition.__call__(train_stats) for condition in self.conditions])


class Plateaus(Condition):
    """Condition that returns ``True`` if"""

    #: Key of train stat to inspect when called.
    key: TrainStatKey

    #: Number of times the value of :attr:`Plateaus.key` has been within
    #: :attr:`Plateaus.rtol` in a row. If this reaches
    #: :attr:`Plateaus.patience`, then the condition is met and
    #: this condition returns ``True``.
    losses: int

    old_value: float

    #: Threshold for :attr:`Plateaus.losses` to reach for the condition
    #: to return ``True``.
    patience: int

    #: Relative tolerance when comparing values of :attr:`Plateaus.key`
    #: betweencalls to determine if the call contributes to
    #: :attr:`Plateaus.losses`.
    rtol: float

    def __init__(
        self, key: TrainStatKey, /, *, patience: int = 5, rtol: float = 1e-3
    ) -> None:
        self.key = key
        self.patience = patience
        self.rtol = rtol
        self.losses = 0
        self.old_value = 0

    def __call__(self, train_stats: TrainStats, /) -> bool:
        new_value = train_stats[self.key]
        if self.old_value and (
            abs(new_value - self.old_value) <= self.rtol * abs(self.old_value)
        ):
            self.losses += 1
        else:
            self.losses = 0
        return self.losses >= self.patience


class HitsLowerBound(Condition):
    def __init__(self, key: TrainStatKey, lower_bound: float, /) -> None:
        self.key = key
        self.lower_bound = lower_bound

    def __call__(self, train_stats: TrainStats, /) -> bool:
        return train_stats[self.key] <= self.lower_bound


class HitsUpperBound(Condition):
    def __init__(self, key: TrainStatKey, upper_bound: float, /) -> None:
        self.key = key
        self.upper_bound = upper_bound

    def __call__(self, train_stats: TrainStats, /) -> bool:
        return train_stats[self.key] >= self.upper_bound
