"""Wrappers around optimizers to handle things like gradient accumulation,
Automatic Mixed Precison (AMP), etc..

"""

from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler


class OptimizerWrapper:
    """Wraps an optimizer with a gradient accumulator depending on the device
    and Automatic Mixed Precision (AMP) flag.

    Args:
        optimizer: Optimizer to accumulate gradients for.
        enable_amp: Whether to enable AMP.
        grad_accumulation_steps: Number of gradient accumulation steps before
            stepping ``optimizer`` and zeroing its gradients.

    """

    #: Number of times :meth:`OptimizerWrapper.step` must be called before stepping
    #: the optimizer and zeroing its gradients.
    grad_accumulation_steps: int

    #: Underlying optimizer that gradients are accumulated for.
    optimizer: optim.Optimizer

    #: CUDA AMP gradient scaler. Does nothing if ``enable_amp`` is ``False``.
    scaler: GradScaler

    #: Counter for number of times :meth:`OptimizerWrapper.step` has been called.
    step_calls: int

    def __init__(
        self,
        optimizer: optim.Optimizer,
        /,
        *,
        enable_amp: bool = False,
        grad_accumulation_steps: int = 1,
    ) -> None:
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.scaler = GradScaler(enabled=enable_amp)  # type: ignore[no-untyped-call]
        self.step_calls = 0

    def step(
        self,
        loss: torch.Tensor,
        params: Iterator[nn.Parameter],
        /,
        *,
        max_grad_norm: float = 5.0,
    ) -> bool:
        """Accumulate gradients using ``loss`` and step ``params`` after
        clipping gradients by ``max_grad_norm`` if the number of gradient
        accumulation steps has been reached.

        Args:
            loss: Loss to call backward on.
            params: Model parameters whose gradients are clipped.
            max_grad_norm: Max gradient magnitude allowed.

        Returned:
            A flag indicating if the optimizer was stepped and the model's
            parameters were updated.

        """
        self.step_calls += 1
        stepped = False
        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        if self.step_calls % self.grad_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(params, max_grad_norm)
            self.scaler.step(self.optimizer)  # type: ignore[no-untyped-call]
            self.scaler.update()  # type: ignore[no-untyped-call]
            self.optimizer.zero_grad()
            stepped = True
        return stepped
