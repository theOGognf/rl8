from abc import abstractmethod
from typing import Any, Generic, Sequence, TypeVar

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import Self

from .._utils import assert_1d_spec
from ..data import DataKeys, Device
from ..nn import MLP, Module, get_activation
from ..specs import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec
from ..views import ViewKind, ViewRequirement

_ObservationSpec = TypeVar("_ObservationSpec", bound=TensorSpec)
_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)


class Model(
    Module[
        [
            TensorDict,
        ],
        TensorDict,
    ]
):
    """Feedforward policy component that processes environment observations into
    a value function approximation and features to be consumed by an
    action distribution for action sampling.

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

    #: Spec defining the forward pass input. Useful for validating the forward
    #: pass and for defining the model as a function of the observation spec.
    observation_spec: TensorSpec

    #: Requirements on how a tensor batch should be preprocessed by the
    #: policy prior to being passed to the forward pass. Useful for handling
    #: sequence shifting or masking so you don't have to.
    #: By default, observations are passed with no shifting.
    #: This should be overwritten in a model's ``__init__`` for custom view
    #: requirements.
    view_requirements: dict[str, ViewRequirement]

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
        self.view_requirements = {DataKeys.OBS: ViewRequirement(shift=0)}

    def apply_view_requirements(
        self, batch: TensorDict, /, *, kind: ViewKind = "last"
    ) -> TensorDict:
        """Apply the model's view requirements, reshaping tensors as-needed.

        This is usually called by the policy that the model is a component
        of, but can be used within the model if the model is deployed without
        the policy or action distribution.

        Args:
            batch: Batch to feed into the policy's underlying model. Expected
                to be of size ``[B, T, ...]`` where ``B`` is the batch dimension,
                and ``T`` is the time or sequence dimension. ``B`` is typically
                the number of parallel environments being sampled for during
                massively parallel training, and ``T`` is typically the number
                of time steps or observations sampled from the environments.
                The ``B`` and ``T`` dimensions are typically combined into one dimension
                during application of the view requirements.
            kind: String indicating the type of view requirements to apply.
                The model's view requirements are applied slightly differently
                depending on the value. Options include:

                    - "last": Apply the view requirements using only the samples
                      necessary to sample for the most recent observations
                      within the ``batch``'s ``T`` dimension.
                    - "all": Sample from ``batch`` using all observations within
                      the ``batch``'s ``T`` dimension. Expand the ``B`` and ``T``
                      dimensions together.

        """
        batch_sizes = {}
        out = {}
        for key, view_requirement in self.view_requirements.items():
            match kind:
                case "all":
                    item = view_requirement.apply_all(key, batch)
                case "last":
                    item = view_requirement.apply_last(key, batch)
            out[key] = item
            B_NEW = item.size(0)
            batch_sizes[key] = B_NEW
        batch_size = next(iter(batch_sizes.values()))
        return TensorDict(out, batch_size=batch_size, device=batch.device)

    @staticmethod
    def default_model_cls(
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
    ) -> type["Model"]:
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
                return DefaultContinuousModel
            case DiscreteTensorSpec():
                return DefaultDiscreteModel
            case _:
                raise TypeError(
                    f"Action spec {action_spec} has no default model support."
                )

    @property
    def device(self) -> Device:
        """Return the device the model is currently on."""
        return next(self.parameters()).device

    @property
    def drop_size(self) -> int:
        """Return the model's drop size (also the drop size for all view
        requirements).

        """
        drop_sizes = {}
        for key, view_requirement in self.view_requirements.items():
            drop_sizes[key] = view_requirement.drop_size
        return next(iter(drop_sizes.values()))

    @abstractmethod
    def forward(self, batch: TensorDict, /) -> TensorDict:
        """Process a batch of tensors and return features to be fed into an
        action distribution.

        Args:
            batch: A tensordict expected to have at least an ``"obs"`` key with any
                tensor spec. The policy that the model is a component of
                processes the batch according to :attr:`Model.view_requirements`
                prior to passing the batch to the forward pass. The tensordict
                must have a 1D batch shape like ``[B, ...]``.

        Returns:
            Features that will be passed to an action distribution with batch
            shape like ``[B, ...]``.

        """

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
        return super().to(device)

    def validate_view_requirements(self) -> None:
        """Helper for validating a model's view requirements.

        Raises:
            RuntimeError: If the model's view requirements result in an
                ambiguous batch size, making training and sampling impossible.

        """
        drop_sizes = {}
        for key, view_requirement in self.view_requirements.items():
            drop_sizes[key] = view_requirement.drop_size
        if len(set(drop_sizes.values())) > 1:
            raise RuntimeError(
                f"""{self} view requirements with drop sizes {drop_sizes}
                result in an ambiguous batch size. It's recommended you:
                    1) use a view requirement method that does not have sample
                        dropping, allowing view requirements with different sizes
                    2) reformulate your model and observation function such
                        that view requirements are not necessary or are
                        handled internal to your environment

                """
            )

    @abstractmethod
    def value_function(self) -> torch.Tensor:
        """Return the value function output for the most recent forward pass.
        Note that a :meth`Model.forward` call has to be performed first before this
        method can return anything.

        This helps prevent extra forward passes from being performed just to
        get a value function output in case the value function and action
        distribution components share parameters.

        """


class GenericModel(Model, Generic[_ObservationSpec, _ActionSpec]):
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


class DefaultContinuousModel(
    GenericModel[UnboundedContinuousTensorSpec, UnboundedContinuousTensorSpec]
):
    """Default model for 1D continuous observations and action spaces."""

    #: Value function estimate set after `forward`.
    _value: None | torch.Tensor

    #: Output head for action log std for a normal distribution.
    action_log_std: nn.Linear

    #: Output head for action mean for a normal distribution.
    action_mean: nn.Linear

    #: Transform observations to inputs for output heads.
    latent_model: nn.Sequential

    #: Value function model, independent of action params.
    vf_model: nn.Sequential

    def __init__(
        self,
        observation_spec: UnboundedContinuousTensorSpec,
        action_spec: UnboundedContinuousTensorSpec,
        /,
        *,
        hiddens: Sequence[int] = (256, 256),
        activation_fn: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__(observation_spec, action_spec)
        self.latent_model = nn.Sequential(
            MLP(
                observation_spec.shape[0],
                hiddens,
                activation_fn=activation_fn,
                bias=bias,
            ),
            get_activation(activation_fn),
        )
        self.action_mean = nn.Linear(hiddens[-1], action_spec.shape[0], bias=True)
        nn.init.uniform_(self.action_mean.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.action_mean.bias)
        self.action_log_std = nn.Linear(hiddens[-1], action_spec.shape[0], bias=True)
        nn.init.uniform_(self.action_log_std.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.action_log_std.bias)
        self.vf_model = nn.Sequential(
            MLP(
                observation_spec.shape[0],
                hiddens,
                activation_fn=activation_fn,
                bias=bias,
            ),
            get_activation(activation_fn),
            nn.Linear(hiddens[-1], 1),
        )
        self._value = None

    def forward(self, batch: TensorDict, /) -> TensorDict:
        obs = batch[DataKeys.OBS]
        latents = self.latent_model(obs)
        action_mean = self.action_mean(latents)
        action_log_std = self.action_log_std(latents)
        self._value = self.vf_model(obs)
        return TensorDict(
            {"mean": action_mean, "log_std": torch.tanh(action_log_std)},
            batch_size=batch.batch_size,
            device=obs.device,
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value


class DefaultDiscreteModel(
    GenericModel[UnboundedContinuousTensorSpec, DiscreteTensorSpec]
):
    """Default model for 1D continuous observations and discrete action spaces."""

    #: Value function estimate set after the forward pass.
    _value: None | torch.Tensor

    #: Transform observations to features for action distributions.
    feature_model: nn.Sequential

    #: Value function model, independent of action params.
    vf_model: nn.Sequential

    def __init__(
        self,
        observation_spec: UnboundedContinuousTensorSpec,
        action_spec: DiscreteTensorSpec,
        /,
        *,
        hiddens: Sequence[int] = (256, 256),
        activation_fn: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__(observation_spec, action_spec)
        self.feature_model = nn.Sequential(
            MLP(
                observation_spec.shape[0],
                hiddens,
                activation_fn=activation_fn,
                bias=bias,
            ),
            get_activation(activation_fn),
        )
        feature_head = nn.Linear(
            hiddens[-1], action_spec.shape[0] * action_spec.space.n
        )
        nn.init.uniform_(feature_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(feature_head.bias)
        self.feature_model.append(feature_head)
        self.vf_model = nn.Sequential(
            MLP(
                observation_spec.shape[0],
                hiddens,
                activation_fn=activation_fn,
                bias=bias,
            ),
            get_activation(activation_fn),
            nn.Linear(hiddens[-1], 1),
        )
        self._value = None

    def forward(self, batch: TensorDict, /) -> TensorDict:
        obs = batch[DataKeys.OBS]
        logits = self.feature_model(obs).reshape(
            -1, self.action_spec.shape[0], self.action_spec.space.n
        )
        self._value = self.vf_model(obs)
        return TensorDict(
            {"logits": logits},
            batch_size=batch.batch_size,
            device=obs.device,
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value
