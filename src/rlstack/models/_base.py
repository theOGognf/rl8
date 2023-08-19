from abc import ABCMeta, abstractmethod
from typing import Any, ParamSpec, TypeVar

import torch
from torchrl.data import TensorSpec
from typing_extensions import Self

from ..data import Device
from ..nn import Module

_P = ParamSpec("_P")
_T = TypeVar("_T")


class GenericModelBase(Module[_P, _T], metaclass=ABCMeta):
    #: Spec defining the outputs of the policy's action distribution that
    #: this model is a component of. Useful for defining the model as a
    #: function of the action spec.
    action_spec: TensorSpec

    #: Model-specific configuration. Passed from the policy and algorithm.
    config: dict[str, Any]

    #: Spec defining the forward pass input. Useful for validating the forward
    #: pass and for defining the model as a function of the observation spec.
    observation_spec: TensorSpec

    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        **config: Any,
    ) -> None:
        super().__init__()
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.config = config

    @property
    def device(self) -> Device:
        """Return the device the model is currently on."""
        return next(self.parameters()).device

    @abstractmethod
    def to(self, device: Device) -> Self:  # type: ignore[override]
        """Helper for changing the device the model is on.

        The specs associated with the model aren't updated with the PyTorch
        module's ``to`` method since they aren't PyTorch modules themselves.

        Args:
            device: Target device.

        Returns:
            The updated model.

        """

    @abstractmethod
    def value_function(self) -> torch.Tensor:
        """Return the value function output for the most recent forward pass.
        Note that a :meth`GenericModelBase.forward` call has to be performed
        first before this method can return anything.

        This helps prevent extra forward passes from being performed just to
        get a value function output in case the value function and action
        distribution components share parameters.

        """
