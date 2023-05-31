from abc import abstractmethod
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import Self

from .._utils import assert_1d_spec
from ..data import DataKeys, Device
from ..nn import Module
from ..specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)

_ObservationSpec = TypeVar("_ObservationSpec", bound=TensorSpec)
_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)


class RecurrentModel(
    Module[
        [TensorDict, TensorDict],
        tuple[TensorDict, TensorDict],
    ]
):
    """Recurrent policy component that processes environment observations and
    recurrent model states into a value function approximation, features
    to be consumed by an action distribution for action sampling, and
    updated recurrent model states to be used for subsequent calls.

    Args:
        observation_spec: Spec defining the forward pass input.
        action_spec: Spec defining the outputs of the policy's action
            distribution that this model is a component of.
        config: Model-specific configuration.

    """

    #: Spec defining the outputs of the policy's action distribution that
    #: this model is a component of. Useful for defining the model as a
    #: function of the action spec.
    action_spec: TensorSpec

    #: Model-specific configuration. Passed from the policy and algorithm.
    config: dict[str, Any]

    #: Spec defining observations part of the forward pass input. Useful for
    #: validating the forward pass and for defining the model as a function of
    #: the observation spec.
    observation_spec: TensorSpec

    #: Spec defining recurrent model states part of the forward pass input
    #: and output. This is expected to be defined in a model's ``__init__``.
    state_spec: CompositeSpec

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

    @staticmethod
    def default_model_cls(
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
    ) -> type["RecurrentModel"]:
        """Return a default model class based on the given observation and
        action specs.

        Args:
            observation_spec: Environment observation spec.
            action_spec: Environment action spec.

        Returns:
            A default model class.

        """
        if not isinstance(observation_spec, UnboundedContinuousTensorSpec):
            raise TypeError(
                f"Observation spec {observation_spec} has no default model support."
            )
        assert_1d_spec(observation_spec)
        assert_1d_spec(action_spec)
        match action_spec:
            case UnboundedContinuousTensorSpec():
                return DefaultContinuousRecurrentModel
            case DiscreteTensorSpec():
                return DefaultDiscreteRecurrentModel
            case _:
                raise TypeError(
                    f"Action spec {action_spec} has no default model support."
                )

    @property
    def device(self) -> Device:
        """Return the device the model is currently on."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(
        self, batch: TensorDict, states: TensorDict, /
    ) -> tuple[TensorDict, TensorDict]:
        """Process a batch of tensors and return features to be fed into an
        action distribution.

        Both input arguments are expected to have a 2D batch shape like
        ``[B, T, ...]`` where ``B`` is the batch number (or typically the
        number of parallel environments) and ``T`` is the sequence length.

        Args:
            batch: A tensordict expected to have at least an ``"obs"`` key with any
                tensor spec.
            states: A tensordict that contains the recurrent states for the
                model and has spec equal to :attr:`RecurrentModel.state_spec`

        Returns:
            Features that will be passed to an action distribution and updated
            recurrent states. The features are expected to have batch shape
            like ``[B * T, ...]`` while the updated recurrent states are
            expected to have batch shape like ``[B, ...]``. In other words,
            the batch and sequence dimension of the input arguments are
            flattened together for the output features while the returned
            recurrent states maintain the original batch dimension but don't
            have a sequence dimension.

        """

    def init_states(self, n: int, /) -> TensorDict:
        """Return initial recurrent states for the model.

        Override this to make your own method for initializing
        recurrent states.

        Args:
            n: Batch size to generate initial recurrent states for.
                This is typically the number of environments being
                stepped in parallel.

        Returns:
            Recurrent model states that initialize a recurrent
            sequence.

        """
        return self.state_spec.zero([n])

    def to(self, device: Device) -> Self:  # type: ignore[override]
        """Helper for changing the device the model is on.

        The specs associated with the model aren't updated with the PyTorch
        module's ``to`` method since they aren't PyTorch modules themselves.

        Args:
            device: Target device.

        Returns:
            The updated model.

        """
        self.observation_spec = self.observation_spec.to(device)
        self.action_spec = self.action_spec.to(device)
        self.state_spec = self.state_spec.to(device)
        return super().to(device)

    @abstractmethod
    def value_function(self) -> torch.Tensor:
        """Return the value function output for the most recent forward pass.
        Note that a :meth`RecurrentModel.forward` call has to be performed
        first before this method can return anything.

        This helps prevent extra forward passes from being performed just to
        get a value function output in case the value function and action
        distribution components share parameters.

        """


class GenericRecurrentModel(RecurrentModel, Generic[_ObservationSpec, _ActionSpec]):
    """Generic model for constructing models from fixed observation and action specs."""

    #: Action space campatible with the model.
    action_spec: _ActionSpec

    #: Observation space compatible with the model.
    observation_spec: _ObservationSpec

    def __init__(
        self,
        observation_spec: _ObservationSpec,
        action_spec: _ActionSpec,
        /,
        **config: Any,
    ) -> None:
        super().__init__(observation_spec, action_spec, **config)


class DefaultContinuousRecurrentModel(
    GenericRecurrentModel[UnboundedContinuousTensorSpec, UnboundedContinuousTensorSpec]
):
    """Default recurrent model for 1D continuous observations and action spaces."""

    #: Value function estimate set after `forward`.
    _value: None | torch.Tensor

    #: Output head for action log std for a normal distribution.
    action_log_std: nn.Linear

    #: Output head for action mean for a normal distribution.
    action_mean: nn.Linear

    #: Transform observations to inputs for output heads.
    lstm: nn.LSTM

    #: Value function model, independent of action params.
    vf_model: nn.Linear

    def __init__(
        self,
        observation_spec: UnboundedContinuousTensorSpec,
        action_spec: UnboundedContinuousTensorSpec,
        /,
        *,
        hidden_size: int = 256,
        num_layers: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(observation_spec, action_spec)
        self.state_spec = CompositeSpec(
            {
                DataKeys.HIDDEN_STATES: UnboundedContinuousTensorSpec(
                    shape=torch.Size([num_layers, hidden_size]),
                    device=action_spec.device,
                ),
                DataKeys.CELL_STATES: UnboundedContinuousTensorSpec(
                    shape=torch.Size([num_layers, hidden_size]),
                    device=action_spec.device,
                ),
            }
        )  # type: ignore[no-untyped-call]
        self.lstm = nn.LSTM(
            observation_spec.shape[0],
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
        )  # type: ignore[no-untyped-call]
        self.action_mean = nn.Linear(hidden_size, action_spec.shape[0], bias=True)
        nn.init.uniform_(self.action_mean.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.action_mean.bias)
        self.action_log_std = nn.Linear(hidden_size, action_spec.shape[0], bias=True)
        nn.init.uniform_(self.action_log_std.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.action_log_std.bias)
        self.vf_model = nn.Linear(hidden_size, 1, bias=bias)
        self._value = None

    def forward(
        self, batch: TensorDict, states: TensorDict, /
    ) -> tuple[TensorDict, TensorDict]:
        obs = batch[DataKeys.OBS]
        h_0 = states[DataKeys.HIDDEN_STATES][:, 0, ...].permute(1, 0, 2).contiguous()
        c_0 = states[DataKeys.CELL_STATES][:, 0, ...].permute(1, 0, 2).contiguous()
        latents, (h_n, c_n) = self.lstm(obs, (h_0, c_0))
        action_mean = self.action_mean(latents).reshape(-1, self.action_spec.shape[0])
        action_log_std = self.action_log_std(latents).reshape(
            -1, self.action_spec.shape[0]
        )
        self._value = self.vf_model(latents).reshape(-1, 1)
        return TensorDict(
            {"mean": action_mean, "log_std": torch.tanh(action_log_std)},
            batch_size=action_mean.size(0),
            device=obs.device,
        ), TensorDict(
            {
                DataKeys.HIDDEN_STATES: h_n.permute(1, 0, 2),
                DataKeys.CELL_STATES: c_n.permute(1, 0, 2),
            },
            batch_size=batch.size(0),
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value


class DefaultDiscreteRecurrentModel(
    GenericRecurrentModel[UnboundedContinuousTensorSpec, DiscreteTensorSpec]
):
    """Default recurrent model for 1D continuous observations and discrete action spaces.
    """

    #: Value function estimate set after the forward pass.
    _value: None | torch.Tensor

    #: Transform observations to features for action distributions.
    feature_head: nn.Linear

    lstm: nn.LSTM

    #: Value function model, independent of action params.
    vf_head: nn.Linear

    def __init__(
        self,
        observation_spec: UnboundedContinuousTensorSpec,
        action_spec: DiscreteTensorSpec,
        /,
        *,
        hidden_size: int = 256,
        num_layers: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(observation_spec, action_spec)
        self.state_spec = CompositeSpec(
            {
                DataKeys.HIDDEN_STATES: UnboundedContinuousTensorSpec(
                    shape=torch.Size([num_layers, hidden_size]),
                    device=action_spec.device,
                ),
                DataKeys.CELL_STATES: UnboundedContinuousTensorSpec(
                    shape=torch.Size([num_layers, hidden_size]),
                    device=action_spec.device,
                ),
            }
        )  # type: ignore[no-untyped-call]
        self.lstm = nn.LSTM(
            observation_spec.shape[0],
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
        )  # type: ignore[no-untyped-call]
        self.feature_head = nn.Linear(
            hidden_size, action_spec.shape[0] * action_spec.space.n, bias=True
        )
        nn.init.uniform_(self.feature_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.feature_head.bias)
        self.vf_head = nn.Linear(hidden_size, 1, bias=bias)
        self._value = None

    def forward(
        self, batch: TensorDict, states: TensorDict, /
    ) -> tuple[TensorDict, TensorDict]:
        obs = batch[DataKeys.OBS]
        h_0 = states[DataKeys.HIDDEN_STATES][:, 0, ...].permute(1, 0, 2).contiguous()
        c_0 = states[DataKeys.CELL_STATES][:, 0, ...].permute(1, 0, 2).contiguous()
        latents, (h_n, c_n) = self.lstm(obs, (h_0, c_0))
        logits = self.feature_head(latents).reshape(
            -1, self.action_spec.shape[0], self.action_spec.space.n
        )
        self._value = self.vf_head(latents).reshape(-1, 1)
        return TensorDict(
            {"logits": logits},
            batch_size=logits.size(0),
            device=obs.device,
        ), TensorDict(
            {
                DataKeys.HIDDEN_STATES: h_n.permute(1, 0, 2),
                DataKeys.CELL_STATES: c_n.permute(1, 0, 2),
            },
            batch_size=batch.size(0),
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value
