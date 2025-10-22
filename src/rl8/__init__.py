"""Top-level package interface."""

from importlib.metadata import PackageNotFoundError, version

from .algorithms import (
    Algorithm,
    AlgorithmConfig,
    RecurrentAlgorithm,
    RecurrentAlgorithmConfig,
)
from .env import Env
from .trainers import RecurrentTrainer, TrainConfig, Trainer

try:
    __version__ = version("rl8")
except PackageNotFoundError:
    pass
