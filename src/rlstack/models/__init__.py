"""Definitions related to parameterizations of policies.

Models are intended to be called with their respective forward pass (like any
other PyTorch module) to get the inputs to the policy's action
distribution (along with other data depending on the type of model being used).
Models are expected to store their value function approximations after each
forward pass in some intermediate attribute so it can be accessed with a
subsequent call to a ``value_function`` method.

Models are largely inspired by RLlib's `model concept`_.

.. _`model concept`: https://github.com/ray-project/ray/blob/master/rllib/models/modelv2.py

"""

from ._feedforward import (
    DefaultContinuousModel,
    DefaultDiscreteModel,
    GenericModel,
    Model,
)
from ._recurrent import (
    DefaultContinuousRecurrentModel,
    DefaultDiscreteRecurrentModel,
    GenericRecurrentModel,
    RecurrentModel,
)

__all__ = [
    "DefaultContinuousModel",
    "DefaultDiscreteModel",
    "GenericModel",
    "Model",
    "DefaultContinuousRecurrentModel",
    "DefaultDiscreteRecurrentModel",
    "GenericRecurrentModel",
    "RecurrentModel",
]
