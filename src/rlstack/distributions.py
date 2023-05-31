from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch
from tensordict import TensorDict

from ._utils import assert_1d_spec
from .specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)

_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)
_FeatureSpec = TypeVar("_FeatureSpec", bound=TensorSpec)
_TorchDistribution = TypeVar(
    "_TorchDistribution", bound=torch.distributions.Distribution
)


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
    #: tensordict, but custom action distributions can return any kind of
    #: tensordict from :attr:`Distribution.model`.
    features: TensorDict

    #: Model from the parent policy also passed to the action distribution.
    #: This is necessary in case the model has components that're only
    #: used for sampling or probability distribution characteristics
    #: computations.
    model: Any

    def __init__(self, features: TensorDict, model: Any, /) -> None:
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
        assert_1d_spec(action_spec)
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

    def sample(self) -> torch.Tensor:
        return self.dist.sample()


class Categorical(
    TorchDistributionWrapper[
        CompositeSpec, torch.distributions.Categorical, DiscreteTensorSpec
    ]
):
    """Wrapper around the PyTorch categorical (i.e., discrete) distribution."""

    def __init__(self, features: TensorDict, model: Any, /) -> None:
        super().__init__(features, model)
        self.dist = torch.distributions.Categorical(logits=features["logits"])  # type: ignore[no-untyped-call]


class Normal(
    TorchDistributionWrapper[
        CompositeSpec, torch.distributions.Normal, UnboundedContinuousTensorSpec
    ]
):
    """Wrapper around the PyTorch normal (i.e., gaussian) distribution."""

    def __init__(self, features: TensorDict, model: Any) -> None:
        super().__init__(features, model)
        self.dist = torch.distributions.Normal(loc=features["mean"], scale=torch.exp(features["log_std"]))  # type: ignore[no-untyped-call]


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
