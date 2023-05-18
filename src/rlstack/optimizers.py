from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim


class OptimizerWrapper:
    def __init__(
        self, optimizer: optim.Optimizer, /, *, grad_accumulation_steps: int = 1
    ) -> None:
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.step_calls = 0

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
