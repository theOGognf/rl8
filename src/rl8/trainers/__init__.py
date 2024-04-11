"""Definitions related to PPO trainers (abstractions over algorithms
and interfaces between those algorithms and other tools).

"""

from ._base import GenericTrainerBase
from ._feedforward import Trainer
from ._recurrent import RecurrentTrainer
from .config import TrainConfig

__all__ = ["GenericTrainerBase", "RecurrentTrainer", "TrainConfig", "Trainer"]
