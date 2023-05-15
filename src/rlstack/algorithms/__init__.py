"""Definitions related to PPO algorithms (data collection and training steps)."""

from ._feedforward import Algorithm
from ._recurrent import RecurrentAlgorithm

__all__ = ["Algorithm", "RecurrentAlgorithm"]
