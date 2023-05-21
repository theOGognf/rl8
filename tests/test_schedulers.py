import torch
import torch.nn as nn
from torch.optim import Adam

from rlstack.schedulers import EntropyScheduler, LRScheduler


def test_interp_entropy_scheduler() -> None:
    entropy_scheduler = EntropyScheduler(
        0.0, schedule=[[0.0, 1.0], [1.0, 2.0]], kind="interp"
    )
    assert entropy_scheduler.step(0.0) == 1.0
    assert entropy_scheduler.step(0.5) == 1.5
    assert entropy_scheduler.step(1.0) == 2.0


def test_interp_lr_scheduler() -> None:
    param = nn.Parameter(torch.tensor([0.0]))
    optimizer = Adam([param])
    lr_scheduler = LRScheduler(
        optimizer, schedule=[[0.0, 1.0], [1.0, 2.0]], kind="interp"
    )
    assert lr_scheduler.step(0.0) == 1.0
    assert lr_scheduler.step(0.5) == 1.5
    assert lr_scheduler.step(1.0) == 2.0


def test_step_entropy_scheduler() -> None:
    entropy_scheduler = EntropyScheduler(
        0.0, schedule=[[0.0, 1.0], [1.0, 2.0]], kind="step"
    )
    assert entropy_scheduler.step(0.0) == 1.0
    assert entropy_scheduler.step(2.0) == 2.0
    assert entropy_scheduler.step(3.0) == 2.0


def test_step_lr_scheduler() -> None:
    param = nn.Parameter(torch.tensor([0.0]))
    optimizer = Adam([param])
    lr_scheduler = LRScheduler(
        optimizer, schedule=[[0.0, 1.0], [1.0, 2.0]], kind="step"
    )
    assert lr_scheduler.step(0.0) == 1.0
    assert lr_scheduler.step(2.0) == 2.0
    assert lr_scheduler.step(3.0) == 2.0
