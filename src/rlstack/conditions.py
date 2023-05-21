"""Definitions for monitoring training metrics and determining whether metrics
achieve some condition (most commonly useful for determining when to stop
training).

"""

from typing import Protocol

from .data import TrainStatKey, TrainStats


class Condition(Protocol):
    """Condition callable that returns ``True`` if a condition is met.

    This is the interface used for early-stopping training.

    """

    def __call__(self, train_stats: TrainStats, /) -> bool:
        """Method to implement that should return ``True`` for forcing training
        within iterations to stop.

        """


class And:
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


class Plateaus:
    """Condition that returns ``True`` if the value being monitored plateaus
    for ``patience`` number of times.

    Args:
        key: Key of train stat to monitor.
        patience: Threshold for :attr:`Plateaus.losses` to reach for the condition
            to return ``True``.
        rtol: Relative tolerance when comparing values of :attr:`Plateaus.key`
            between calls to determine if the call contributes to
            :attr:`Plateaus.losses`.

    """

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
    #: between calls to determine if the call contributes to
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
        if abs(new_value - self.old_value) <= self.rtol * abs(self.old_value):
            self.losses += 1
        else:
            self.losses = 0
        self.old_value = new_value
        return self.losses >= self.patience


class HitsLowerBound:
    """Condition that returns ``True`` if the value being monitored hits a
    lower bound value.

    Args:
        key: Key of train stat to monitor.
        lower_bound: Minimum threshold for the value of ``key`` to reach before
            this condition returns ``True`` when called.

    """

    #: Key of train stat to inspect when called.
    key: TrainStatKey

    #: Minimum threshold for the value of ``key`` to reach before
    #: this condition returns ``True`` when called.
    lower_bound: float

    def __init__(self, key: TrainStatKey, lower_bound: float, /) -> None:
        self.key = key
        self.lower_bound = lower_bound

    def __call__(self, train_stats: TrainStats, /) -> bool:
        return train_stats[self.key] <= self.lower_bound


class HitsUpperBound:
    """Condition that returns ``True`` if the value being monitored hits an
    upper bound value.

    Args:
        key: Key of train stat to monitor.
        upper_bound: Maximum threshold for the value of ``key`` to reach before
            this condition returns ``True`` when called.

    """

    #: Key of train stat to inspect when called.
    key: TrainStatKey

    #: Maximum threshold for the value of ``key`` to reach before
    #: this condition returns ``True`` when called.
    upper_bound: float

    def __init__(self, key: TrainStatKey, upper_bound: float, /) -> None:
        self.key = key
        self.upper_bound = upper_bound

    def __call__(self, train_stats: TrainStats, /) -> bool:
        return train_stats[self.key] >= self.upper_bound
