"""Schedulers for scheduling values, learning rates, and entropy."""

from typing import Literal, Protocol

import numpy as np
import torch.optim as optim

ScheduleKind = Literal["interp", "step"]


class Scheduler(Protocol):
    """Scheduler protocol for returning a value according to environment
    sample count.

    """

    def step(self, count: int, /) -> float:
        """Return the value associated with the schedule for ``count`` environment
        transitions.

        """


class ConstantScheduler:
    """Scheduler that outputs a constant value.

    This is the default scheduler when a schedule type is not provided.

    Args:
        value: Constant value to output.

    """

    #: Constant schedule value.
    value: float

    def __init__(self, value: float, /) -> None:
        super().__init__()
        self.value = value

    def step(self, _: int, /) -> float:
        return self.value


class InterpScheduler:
    """Scheduler that interpolates between steps to new values when the
    number of environment samples exceeds a threshold.

    Args:
        schedule: List of tuples where the first element of the tuple is the
            number of environment transitions needed to trigger a step and
            the second element of the tuple is the value to step to when
            a step is triggered.

    """

    #: Number of environment transitions needed to trigger the next value
    #: step in ``y``.
    x: list[int]

    #: Value to step the schedule to when a the number of environment
    #: transitions exceeds the corresponding element in ``x``.
    y: list[float]

    def __init__(self, schedule: list[tuple[int, float]], /) -> None:
        super().__init__()
        if schedule[0][0]:
            raise ValueError(
                f"{self.__class__.__name__} `schedule` arg's first "
                "step value (i.e., `schedule[0][0]`) must be `0` to "
                "indicate the scheduler's initial value."
            )
        self.x = []
        self.y = []
        for x, y in schedule:
            self.x.append(x)
            self.y.append(y)

    def step(self, count: int, /) -> float:
        return float(np.interp(count, self.x, self.y))


class StepScheduler:
    """Scheduler that steps to a new value when the number of environment
    samples exceeds a threshold.

    This is the default scheduler when a schedule is provided.

    Args:
        schedule: List of tuples where the first element of the tuple is the
            number of environment transitions needed to trigger a step and
            the second element of the tuple is the value to step to when
            a step is triggered.

    """

    #: List of tuples where the first element of the tuple is the
    #: number of environment transitions needed to trigger a step and
    #: the second element of the tuple is the value to step to when
    #: a step is triggered.
    schedule: list[tuple[int, float]]

    def __init__(self, schedule: list[tuple[int, float]], /) -> None:
        super().__init__()
        if schedule[0][0]:
            raise ValueError(
                f"{self.__class__.__name__} `schedule` arg's first "
                "step value (i.e., `schedule[0][0]`) must be `0` to "
                "indicate the scheduler's initial value."
            )
        self.schedule = schedule

    def step(self, count: int, /) -> float:
        value = 0.0
        for t, v in self.schedule:
            if count >= t:
                value = v
        return value


class EntropyScheduler:
    """Entropy scheduler for scheduling entropy coefficients based on environment
    transition counts during learning.

    Args:
        coeff: Entropy coefficient value. This value is ignored if a
            ``schedule`` is provided.
        schedule: Optional schedule that overrides ``coeff``. This determines
            values of `coeff` according to the number of environment
            transitions experienced during learning.
        kind: Kind of scheduler to use. Options include:

            - "step": jump to values and hold until a new environment transition
                count is reached.
            - "interp": jump to values like "step", but interpolate between the
                current value and the next value.

    """

    #: Current entropy coefficient value.
    coeff: float

    #: Backend value scheduler used. The type depends on if a ``schedule`` arg is
    #: provided and ``kind``.
    scheduler: Scheduler

    def __init__(
        self,
        coeff: float,
        /,
        *,
        schedule: None | list[tuple[int, float]] = None,
        kind: ScheduleKind = "step",
    ) -> None:
        if schedule is None:
            self.scheduler = ConstantScheduler(coeff)
        else:
            match kind:
                case "interp":
                    self.scheduler = InterpScheduler(schedule)
                case "step":
                    self.scheduler = StepScheduler(schedule)
                case _:
                    raise ValueError(
                        f"Entropy scheduler only supports kinds `interp` and `step`."
                    )
        self.coeff = self.step(0)

    def step(self, count: int, /) -> float:
        self.coeff = self.scheduler.step(count)
        return self.coeff


class LRScheduler:
    """Learning rate scheduler for scheduling optimizer learning rates based on
    environment transition counts during learning.

    Args:
        optimizer: Optimizer to update with each :meth:`LRScheduler.step`.
        schedule: Optional schedule that overrides the optimizer's learning rate.
            This determines values of the learning rate according to the
            number of environment transitions experienced during learning.
        kind: Kind of scheduler to use. Options include:

            - "step": jump to values and hold until a new environment transition
                count is reached.
            - "interp": jump to values like "step", but interpolate between the
                current value and the next value.

    """

    #: Current learning rate according to the schedule. ``0`` until
    #: :meth:`LRScheduler.step` is called for the first time.
    coeff: float

    #: Backend optimizer whose learning rate is updated in-place.
    optimizer: optim.Optimizer

    #: Backend value scheduler used. The type depends on if a ``schedule`` arg is
    #: provided and ``kind``'s value.
    scheduler: Scheduler

    def __init__(
        self,
        optimizer: optim.Optimizer,
        /,
        *,
        schedule: None | list[tuple[int, float]] = None,
        kind: ScheduleKind = "step",
    ) -> None:
        self.optimizer = optimizer
        if schedule is None:
            self.scheduler = ConstantScheduler(0.0)
        else:
            match kind:
                case "interp":
                    self.scheduler = InterpScheduler(schedule)
                case "step":
                    self.scheduler = StepScheduler(schedule)
                case _:
                    raise ValueError(
                        f"Learning rate scheduler only supports "
                        f"kinds `interp` and `step`."
                    )
        self.coeff = self.step(0)

    def step(self, count: int, /) -> float:
        self.coeff = self.scheduler.step(count)
        if not isinstance(self.scheduler, ConstantScheduler):
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.coeff
        return self.coeff
