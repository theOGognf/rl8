from typing import Any, Generic, TypeVar

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


class GenericPolicyBase(Generic[_Model]):
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

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self
