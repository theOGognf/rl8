from typing import Protocol

from .data import TrainStatKey, TrainStats


class Condition(Protocol):
    def eval(self, train_stats: TrainStats, /) -> bool:
        ...


class And(Condition):
    def __init__(self, conditions: list[Condition], /) -> None:
        self.conditions = conditions

    def eval(self, train_stats: TrainStats, /) -> bool:
        return all([condition.eval(train_stats) for condition in self.conditions])


class Plateaus(Condition):
    #: Key of train stat to inspect in :meth:`Plateaus.eval`.
    key: TrainStatKey

    #: Number of times the value of ``key`` has been within :attr:`Plateaus.rtol`
    #: in a row. If this reaches :attr:`Plateaus.patience`, then the condition
    #: is met and :meth:`Plateaus.eval` returns ``True``.
    losses: int

    def __init__(
        self, key: TrainStatKey, /, *, patience: int = 5, rtol: float = 1e-3
    ) -> None:
        self.key = key
        self.patience = patience
        self.rtol = rtol
        self.losses = 0
        self.old_value = 0

    def eval(self, train_stats: TrainStats, /) -> bool:
        if self.key in train_stats:
            new_value = train_stats[self.key]
            if self.old_value and (
                abs(new_value - self.old_value) <= self.rtol * abs(self.old_value)
            ):
                self.losses += 1
            elif not new_value:
                self.losses += 1
            else:
                self.losses = 0
        return self.losses >= self.patience


class HitsLowerBound(Condition):
    def __init__(self, key: TrainStatKey, lower_bound: float, /) -> None:
        self.key = key
        self.lower_bound = lower_bound

    def eval(self, train_stats: TrainStats, /) -> bool:
        if self.key in train_stats:
            return train_stats[self.key] <= self.lower_bound
        return False


class HitsUpperBound(Condition):
    def __init__(self, key: TrainStatKey, upper_bound: float, /) -> None:
        self.key = key
        self.upper_bound = upper_bound

    def eval(self, train_stats: TrainStats, /) -> bool:
        if self.key in train_stats:
            return train_stats[self.key] >= self.upper_bound
        return False
