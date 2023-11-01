"""Definitions related to PPO trainers (abstractions over algorithms
and interfaces between those algorithms and other tools).

"""

from ._base import GenericTrainerBase
from ._feedforward import Trainer
from ._recurrent import RecurrentTrainer

__all__ = ["GenericTrainerBase", "Trainer", "RecurrentTrainer"]
