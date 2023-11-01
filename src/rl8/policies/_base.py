import os
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

import mlflow
from torchrl.data import TensorSpec
from typing_extensions import Self

from ..data import Device
from ..distributions import Distribution
from ..models import GenericModelBase

_Model = TypeVar(
    "_Model",
    bound=GenericModelBase[
        [
            Any,
        ],
        Any,
    ],
)


class GenericPolicyBase(Generic[_Model], metaclass=ABCMeta):
    #: Underlying policy action distribution that's parameterized by
    #: features produced by :attr:`GenericPolicyBase.model`.
    distribution_cls: type[Distribution]

    #: Underlying policy model that processes environment observations
    #: into a value function approximation and into features to be
    #: consumed by an action distribution for action sampling.
    model: _Model

    @property
    def action_spec(self) -> TensorSpec:
        """Return the action spec used for constructing the model."""
        return self.model.action_spec

    @property
    def device(self) -> Device:
        """Return the device the policy's model is on."""
        return self.model.device

    @property
    def observation_spec(self) -> TensorSpec:
        """Return the observation spec used for constructing the model."""
        return self.model.observation_spec

    @abstractmethod
    def save(self, path: str | os.PathLike[str], /) -> mlflow.pyfunc.PythonModel:
        """Save the policy by cloud pickling it to ``path`` and returning
        the interface used for deploying it with MLflow.

        This method is only defined to expose a common interface between
        different algorithms. This is by no means the only way
        to save a policy and isn't even the recommended way to save
        a policy.

        """

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self
