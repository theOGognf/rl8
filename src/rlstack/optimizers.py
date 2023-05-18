"""Wrappers around optimizers to handle things like gradient accumulation,
Automatic Mixed Precison (AMP), etc..

"""

from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

from .data import Device


class Accumulator(ABC):
    """Abstract class for optimizers that accumulates gradients.

    Args:
        optimizer: Optimizer that the accumulator uses when accumulating
            gradients.
        grad_accumulation_steps: Number of gradient accumulation steps before
            stepping ``optimizer`` and zeroing its gradients.

    """

    #: Number of times :meth:`Accumulator.step` must be called before stepping
    #: the optimizer and zeroing its gradients.
    grad_accumulation_steps: int

    #: Underlying optimizer that gradients are accumulated for.
    optimizer: optim.Optimizer

    #: Counter for number of times :meth:`Accumulator.step` has been called.
    step_calls: int

    def __init__(
        self, optimizer: optim.Optimizer, /, *, grad_accumulation_steps: int = 1
    ) -> None:
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.step_calls = 0

    @abstractmethod
    def step(
        self,
        loss: torch.Tensor,
        params: Iterator[nn.Parameter],
        /,
        *,
        max_grad_norm: float = 5.0,
    ) -> None:
        """Somehow call backward on ``loss`` and optionally clip the gradients
        of ``params`` while accumulating gradients.

        """


class ScaledAccumulator(Accumulator):
    """This accumulator accumulates gradients and takes an optimization
    step when ``grad_accumulation_steps`` is hit and scales gradients
    for Automatic Mixed Precision (AMP) usage with CUDA devices.

    """

    #: CUDA AMP gradient scaler.
    scaler: GradScaler

    def __init__(
        self, optimizer: optim.Optimizer, /, *, grad_accumulation_steps: int = 1
    ) -> None:
        super().__init__(optimizer, grad_accumulation_steps=grad_accumulation_steps)
        self.scaler = GradScaler()  # type: ignore[no-untyped-call]

    def step(
        self,
        loss: torch.Tensor,
        params: Iterator[nn.Parameter],
        /,
        *,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        if self.step_calls % self.grad_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(params, max_grad_norm)
            self.scaler.step(self.optimizer)  # type: ignore[no-untyped-call]
            self.scaler.update()  # type: ignore[no-untyped-call]
            self.optimizer.zero_grad()
        self.step_calls += 1


class SimpleAccumulator(Accumulator):
    """This accumulator simply accumulates gradients and takes an optimization
    step when ``grad_accumulation_steps`` is hit.

    """

    def step(
        self,
        loss: torch.Tensor,
        params: Iterator[nn.Parameter],
        /,
        *,
        max_grad_norm: float = 5.0,
    ) -> None:
        loss.backward()  # type: ignore[no-untyped-call]
        if self.step_calls % self.grad_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(params, max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.step_calls += 1


class OptimizerWrapper:
    """Wraps an optimizer with a gradient accumulator depending on the device
    and Automatic Mixed Precision (AMP) flag.

    Args:
        optimizer: Optimizer to accumulate gradients for.
        device: Device gradients will be computed on.
        enable_amp: Whether to enable AMP.
        grad_accumulation_steps: Number of gradient accumulation steps before
            stepping ``optimizer`` and zeroing its gradients.

    """

    #: Underlying accumulator that accumulates gradients and controls when
    #: and how to step the optimizer.
    accumulator: Accumulator

    def __init__(
        self,
        optimizer: optim.Optimizer,
        /,
        *,
        device: Device = "cpu",
        enable_amp: bool = False,
        grad_accumulation_steps: int = 1,
    ) -> None:
        accumulator_cls = (
            ScaledAccumulator if device == "cuda" and enable_amp else SimpleAccumulator
        )
        self.accumulator = accumulator_cls(
            optimizer, grad_accumulation_steps=grad_accumulation_steps
        )

    @property
    def optimizer(self) -> optim.Optimizer:
        """Return the underlying optimizer."""
        return self.accumulator.optimizer

    def step(
        self,
        loss: torch.Tensor,
        params: Iterator[nn.Parameter],
        /,
        *,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.accumulator.step(loss, params, max_grad_norm=max_grad_norm)
