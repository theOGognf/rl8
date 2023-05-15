"""Definitions related to the union of models and action distributions.

This is the main definition used by training algorithms for sampling
from models and action distributions. It's recommended to use this interface
when deploying a policy or model such that the action distribution
is always paired with the model and transformations required for model
inference are properly handled.

"""

from ._feedforward import Policy
from ._recurrent import RecurrentPolicy

__all__ = ["Policy", "RecurrentPolicy"]
