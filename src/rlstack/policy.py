"""Definitions regarding the union of a model and an action distribution."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generic, Sequence, TypeVar

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import Self

from .data import DataKeys, Device
from .nn import MLP, Module, get_activation
from .specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from .view import ViewKind, ViewRequirement

_ObservationSpec = TypeVar("_ObservationSpec", bound=TensorSpec)
_FeatureSpec = TypeVar("_FeatureSpec", bound=TensorSpec)
_TorchDistribution = TypeVar(
    "_TorchDistribution", bound=torch.distributions.Distribution
)
_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)


class Model(
    Module[
        [
            TensorDict,
        ],
        TensorDict,
    ]
):
    """Policy component that processes environment observations into
    a value function approximation and features to be consumed by an
    action distribution for action sampling.

    This definition is largely inspired by RLlib's `model concept`_.

    The model is intended to be called with the forward pass (like any
    other PyTorch module) to get the inputs to the policy's action
    distribution. It's expected that the value function approximation
    is stored after each forward pass in some intermediate attribute
    and can be accessed with a subsequent call to :meth:`Model.value_function`.

    Args:
        observation_spec: Spec defining the forward pass input.
        action_spec: Spec defining the outputs of the policy's action
            distribution that this model is a component of.
        config: Model-specific configuration.

    .. _`model concept`: https://github.com/ray-project/ray/blob/master/rllib/models/modelv2.py

    """

    #: Spec defining the outputs of the policy's action distribution that
    #: this model is a component of. Useful for defining the model as a
    #: function of the action spec.
    action_spec: TensorSpec

    #: Model-specific configuration. Passed from the policy and algorithm.
    config: dict[str, Any]

    #: Spec defining the forward pass output. Useful for passing inputs to an
    #: action distribution or stroing values in the replay buffer. Defaults
    #: to `action_spec`. This should be overwritten in a model's ``__init__``
    #: for models that feed into custom action distributions .
    feature_spec: TensorSpec

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
        self.feature_spec = self.default_feature_spec(action_spec)
        self.view_requirements = {DataKeys.OBS: ViewRequirement(DataKeys.OBS, shift=0)}

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
                    item = view_requirement.apply_all(batch)
                case "last":
                    item = view_requirement.apply_last(batch)
            out[key] = item
            B_NEW = item.size(0)
            batch_sizes[key] = B_NEW
        batch_size = next(iter(batch_sizes.values()))
        return TensorDict(out, batch_size=batch_size, device=batch.device)

    @property
    def burn_size(self) -> int:
        """Return the model's burn size (also the burn size for all view
        requirements.

        """
        burn_sizes = {}
        for key, view_requirement in self.view_requirements.items():
            burn_sizes[key] = view_requirement.burn_size
        return next(iter(burn_sizes.values()))

    @staticmethod
    def default_feature_spec(action_spec: TensorSpec, /) -> TensorSpec:
        """Return a default feature spec given an action spec.

        Useful for defining feature specs for simple and common action
        specs. Custom models with complex action specs should define
        their own custom feature specs as an attribute.

        Args:
            action_spec: Spec defining the outputs of the policy's action
                distribution that this model is a component of. Typically
                passed into the model's ``__init__``.

        Returns:
            A spec defining the inputs to the policy's action distribution.
            For simple distributions (e.g., categorical or diagonal gaussian),
            this returns a spec defining the inputs to those distributions
            (e.g., logits and mean/scales, respectively). For complex
            distributions, this returns a copy of the action spec and the model
            is expected to assign the correct feature spec within its own
            ``__init__``.

        """
        match action_spec:
            case DiscreteTensorSpec():
                return Categorical.required_feature_spec(action_spec)
            case UnboundedContinuousTensorSpec():
                return Normal.required_feature_spec(action_spec)
            case _:
                return deepcopy(action_spec)

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
        _assert_1d_spec(observation_spec)
        _assert_1d_spec(action_spec)
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

    @abstractmethod
    def forward(self, batch: TensorDict, /) -> TensorDict:
        """Process a batch of tensors and return features to be fed into an
        action distribution.

        Args:
            batch: A tensordict expected to have at least an ``"obs"`` key with any
                tensor spec. The policy that the model is a component of
                processes the batch according to :attr:`Model.view_requirements`
                prior to passing the batch to the forward pass.

        Returns:
            Features that will be passed to an action distribution.

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
        self.feature_spec = self.feature_spec.to(device)
        return super().to(device)

    def validate_view_requirements(self) -> None:
        """Helper for validating a model's view requirements.

        Raises:
            RuntimeError: If the model's view requirements result in an
                ambiguous batch size, making training and sampling impossible.

        """
        burn_sizes = {}
        for key, view_requirement in self.view_requirements.items():
            burn_sizes[key] = view_requirement.burn_size
        if len(set(burn_sizes.values())) > 1:
            raise RuntimeError(
                f"""{self} view requirements with burn sizes {burn_sizes}
                result in an ambiguous batch size. It's recommended you:
                    1) use a view requirement method that does not have sample
                        burn, allowing view requirements with different sizes
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


def _assert_1d_spec(spec: TensorSpec, /) -> None:
    """Check if the spec is valid for default models and distributions
    by asserting that it's 1D.

    Args:
        spec: Observation or action spec when using default models or
            distributions.

    Raises:
        AssertionError: If ``spec`` is not 1D.

    """
    assert spec.shape.numel() == 1, (
        "Default models and distributions do not support tensor specs "
        "that aren't 1D. Tensor specs must have shape ``[N]`` "
        "(where ``N`` is the number of independent elements) to be "
        "compatible with default models and distributions."
    )


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
        obs = batch["obs"]
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
        obs = batch["obs"]
        features = self.feature_model(obs)
        self._value = self.vf_model(obs)
        return TensorDict(
            {
                "logits": features.reshape(
                    -1, self.action_spec.shape[0], self.action_spec.space.n
                )
            },
            batch_size=batch.batch_size,
            device=obs.device,
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value


class Distribution(ABC):
    """Policy component that defines a probability distribution over a
    feature set from a model.

    This definition is largely inspired by RLlib's `action distribution`_.

    Most commonly, the feature set is a single vector of logits or log
    probabilities used for defining and sampling from the probability
    distribution. Custom probabiltiy distributions, however, are not
    constrained to just a single vector.

    Args:
        features: Features from ``model``'s forward pass.
        model: Model for parameterizing the probability distribution.

    .. _`action distribution`: https://github.com/ray-project/ray/blob/master/rllib/models/action_dist.py

    """

    #: Features from :attr:`Distribution.model` forward pass. Simple action
    #: distributions expect one field and corresponding tensor in the
    #: tensor dict, but custom action distributions can return any kind of
    #: tensor dict from :attr:`Distribution.model`.
    features: TensorDict

    #: Model from the parent policy also passed to the action distribution.
    #: This is necessary in case the model has components that're only
    #: used for sampling or probability distribution characteristics
    #: computations.
    model: Model

    def __init__(self, features: TensorDict, model: Model, /) -> None:
        super().__init__()
        self.features = features
        self.model = model

    @staticmethod
    def default_dist_cls(action_spec: TensorSpec, /) -> type["Distribution"]:
        """Return a default distribution given an action spec.

        Args:
            action_spec: Spec defining required environment inputs.

        Returns:
            A distribution for simple, supported action specs.

        """
        _assert_1d_spec(action_spec)
        match action_spec:
            case DiscreteTensorSpec():
                return Categorical
            case UnboundedContinuousTensorSpec():
                return Normal
            case _:
                raise TypeError(
                    f"Action spec {action_spec} has no default distribution support."
                )

    @abstractmethod
    def deterministic_sample(self) -> torch.Tensor | TensorDict:
        """Draw a deterministic sample from the probability distribution."""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """Compute the probability distribution's entropy (a measurement
        of randomness).

        """

    @abstractmethod
    def logp(self, samples: torch.Tensor | TensorDict) -> torch.Tensor:
        """Compute the log probability of sampling `samples` from the probability
        distribution.

        """

    @abstractmethod
    def sample(self) -> torch.Tensor | TensorDict:
        """Draw a stochastic sample from the probability distribution."""


class TorchDistributionWrapper(
    Distribution, Generic[_FeatureSpec, _TorchDistribution, _ActionSpec]
):
    """Wrapper class for PyTorch distributions.

    This is inspired by `RLlib`_.

    .. _`RLlib`: https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_action_dist.py

    """

    #: Underlying PyTorch distribution.
    dist: _TorchDistribution

    def deterministic_sample(self) -> torch.Tensor:
        return self.dist.mode

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1, keepdim=True)

    def logp(self, samples: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(samples).sum(-1, keepdim=True)

    @staticmethod
    @abstractmethod
    def required_feature_spec(action_spec: _ActionSpec, /) -> _FeatureSpec:
        """Define feature spec requirements for the distribution given an
        action spec.

        """

    def sample(self) -> torch.Tensor:
        return self.dist.sample()


class Categorical(
    TorchDistributionWrapper[
        CompositeSpec, torch.distributions.Categorical, DiscreteTensorSpec
    ]
):
    """Wrapper around the PyTorch categorical (i.e., discrete) distribution."""

    def __init__(self, features: TensorDict, model: Model, /) -> None:
        super().__init__(features, model)
        self.dist = torch.distributions.Categorical(logits=features["logits"])  # type: ignore[no-untyped-call]

    @staticmethod
    def required_feature_spec(action_spec: DiscreteTensorSpec, /) -> CompositeSpec:
        return CompositeSpec(
            logits=UnboundedContinuousTensorSpec(
                shape=torch.Size([action_spec.shape[0], action_spec.space.n]),
                device=action_spec.device,
            )
        )  # type: ignore[no-untyped-call]


class Normal(
    TorchDistributionWrapper[
        CompositeSpec, torch.distributions.Normal, UnboundedContinuousTensorSpec
    ]
):
    """Wrapper around the PyTorch normal (i.e., gaussian) distribution."""

    def __init__(self, features: TensorDict, model: Model) -> None:
        super().__init__(features, model)
        self.dist = torch.distributions.Normal(loc=features["mean"], scale=torch.exp(features["log_std"]))  # type: ignore[no-untyped-call]

    @staticmethod
    def required_feature_spec(
        action_spec: UnboundedContinuousTensorSpec, /
    ) -> CompositeSpec:
        return CompositeSpec(
            mean=UnboundedContinuousTensorSpec(
                shape=action_spec.shape, device=action_spec.device
            ),
            log_std=UnboundedContinuousTensorSpec(
                shape=action_spec.shape, device=action_spec.device
            ),
        )  # type: ignore[no-untyped-call]


class SquashedNormal(Normal):
    """Squashed normal distribution such that samples are always within [-1, 1]."""

    def deterministic_sample(self) -> torch.Tensor:
        return super().deterministic_sample().tanh()

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError(
            f"Entropy isn't defined for {self.__class__.__name__}. Set the"
            " entropy coefficient to `0` to avoid this error during training."
        )

    def logp(self, samples: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(samples.dtype).eps
        clipped_samples = samples.clamp(min=-1 + eps, max=1 - eps)
        inverted_samples = 0.5 * (clipped_samples.log1p() - (-clipped_samples).log1p())
        logp = torch.clamp(self.dist.log_prob(inverted_samples), min=-100, max=100).sum(  # type: ignore[no-untyped-call]
            -1, keepdim=True
        )
        logp -= torch.sum(torch.log(1 - samples**2 + eps), dim=-1, keepdim=True)
        return logp

    def sample(self) -> torch.Tensor:
        return super().sample().tanh()


class Policy:
    """The union of a model and an action distribution.

    This is the main definition used by training algorithms for sampling
    and other data aggregations. It's recommended to use this interface
    when deploying a policy or model such that the action distribution
    is always paired with the model and the model's view requirements are
    always respected.

    Args:
        observation_spec: Spec defining observations from the environment
            and inputs to the model's forward pass.
        action_spec: Spec defining the action distribution's outputs
            and the inputs to the environment.
        model: Model instance to use. Mutually exclusive with ``model_cls``.
        model_cls: Model class to use.
        model_config: Model class args.
        dist_cls: Action distribution class.

    """

    #: Underlying policy action distribution that's parameterized by
    #: features produced by :attr:`Policy.model`.
    dist_cls: type[Distribution]

    #: Underlying policy model that processes environment observations
    #: into a value function approximation and into features to be
    #: consumed by an action distribution for action sampling.
    model: Model

    #: Model kwarg overrides when instantiating the model.
    model_config: dict[str, Any]

    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        *,
        model: None | Model = None,
        model_cls: None | type[Model] = None,
        model_config: None | dict[str, Any] = None,
        dist_cls: None | type[Distribution] = None,
        device: Device = "cpu",
    ) -> None:
        self.model_config = model_config or {}
        if model and model_cls:
            raise ValueError(
                "`model` and `model_cls` args are mutually exclusive."
                "Provide one or the other, but not both."
            )
        if model is None:
            model_cls = model_cls or Model.default_model_cls(
                observation_spec, action_spec
            )
            self.model = model_cls(observation_spec, action_spec, **self.model_config)
        else:
            self.model = model
        self.model = self.model.to(device)
        self.dist_cls = dist_cls or Distribution.default_dist_cls(action_spec)

    @property
    def action_spec(self) -> TensorSpec:
        """Return the action spec used for constructing the model."""
        return self.model.action_spec

    @property
    def device(self) -> Device:
        """Return the device the policy's model is on."""
        return next(self.model.parameters()).device

    @property
    def feature_spec(self) -> TensorSpec:
        """Return the feature spec defined in the model."""
        return self.model.feature_spec

    @property
    def observation_spec(self) -> TensorSpec:
        """Return the observation spec used for constructing the model."""
        return self.model.observation_spec

    def sample(
        self,
        batch: TensorDict,
        /,
        *,
        kind: ViewKind = "last",
        deterministic: bool = False,
        inplace: bool = False,
        requires_grad: bool = False,
        return_actions: bool = True,
        return_logp: bool = False,
        return_values: bool = False,
        return_views: bool = False,
    ) -> TensorDict:
        """Use ``batch`` to sample from the policy, sampling actions from
        the model and optionally sampling additional values often used for
        training and analysis.

        Args:
            batch: Batch to feed into the policy's underlying model. Expected
                to be of size ``[B, T, ...]`` where ``B`` is the batch dimension,
                and ``T`` is the time or sequence dimension. ``B`` is typically
                the number of parallel environments being sampled for during
                massively parallel training, and T is typically the number
                of time steps or observations sampled from the environments.
                The ``B`` and ``T`` dimensions are typically combined into one dimension
                during batch preprocessing according to the model's view
                requirements.
            kind: String indicating the type of sample to perform. The model's
                view requirements handles preprocessing slightly differently
                depending on the value. Options include:

                    - "last": Sample from ``batch`` using only the samples
                        necessary to sample for the most recent observations
                        within the ``batch``'s ``T`` dimension.
                    - "all": Sample from ``batch`` using all observations within
                        the ``batch``'s ``T`` dimension.

            deterministic: Whether to sample from the policy deterministically
                (the actions are always the same for the same inputs) or
                stochastically (there is a randomness to the policy's actions).
            inplace: Whether to store policy outputs in the given ``batch``
                tensor dict. Otherwise, create a separate tensor dict that
                will only contain policy outputs.
            requires_grad: Whether to enable gradients for the underlying
                model during forward passes. This should only be enabled during
                a training loop or when requiring gradients for explainability
                or other analysis reasons.
            return_actions: Whether to sample the policy's action distribution
                and return the sampled actions.
            return_logp: Whether to return the log probability of taking the
                sampled actions. Often enabled during a training loop for
                aggregating training data a bit more efficiently.
            return_values: Whether to return the value function approximation
                in the given observations. Often enabled during a training
                loop for aggregating training data a bit more efficiently.
            return_views: Whether to return the observation view requirements
                in the output batch. Even if this flag is enabled, new views
                are only returned if the views are not already present in
                the output batch (i.e., if `inplace` is ``True`` and the views
                are already in the ``batch``, then the returned batch will just
                contain the original views).

        Returns:
            A tensor dict containing AT LEAST actions sampled from the policy.

        """
        if DataKeys.VIEWS in batch.keys():
            in_batch = batch[DataKeys.VIEWS]
        else:
            in_batch = self.model.apply_view_requirements(batch, kind=kind)

        # This is the same mechanism within `torch.no_grad`
        # for enabling/disabling gradients.
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(requires_grad)

        features = self.model(in_batch)

        # Store required outputs and get/store optional outputs.
        out = (
            batch
            if inplace
            else TensorDict({}, batch_size=in_batch.batch_size, device=batch.device)
        )
        out[DataKeys.FEATURES] = features
        if return_actions:
            dist = self.dist_cls(features, self.model)
            actions = dist.deterministic_sample() if deterministic else dist.sample()
            out[DataKeys.ACTIONS] = actions
            if return_logp:
                out[DataKeys.LOGP] = dist.logp(actions)
        if return_values:
            out[DataKeys.VALUES] = self.model.value_function()
        if return_views:
            out[DataKeys.VIEWS] = in_batch

        torch.set_grad_enabled(prev)
        return out

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self
