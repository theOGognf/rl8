"""Definitions regarding the union of a model and an action distribution."""

from typing import Any

import torch
from tensordict import TensorDict
from typing_extensions import Self

from ..data import DataKeys, Device
from ..distributions import Distribution
from ..models import RecurrentModel
from ..specs import TensorSpec


class RecurrentPolicy:
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
    model: RecurrentModel

    #: Model kwarg overrides when instantiating the model.
    model_config: dict[str, Any]

    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        *,
        model: None | RecurrentModel = None,
        model_cls: None | type[RecurrentModel] = None,
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
            model_cls = model_cls or RecurrentModel.default_model_cls(
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
        states: TensorDict,
        /,
        *,
        deterministic: bool = False,
        inplace: bool = False,
        keepdim: bool = False,
        requires_grad: bool = False,
        return_actions: bool = True,
        return_logp: bool = False,
        return_values: bool = False,
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

        Returns:
            A tensor dict containing AT LEAST actions sampled from the policy.

        """
        # This is the same mechanism within `torch.no_grad`
        # for enabling/disabling gradients.
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(requires_grad)

        features, out_states = self.model(batch, states)

        # Store required outputs and get/store optional outputs.
        out = (
            batch
            if inplace
            else TensorDict({}, batch_size=features.batch_size, device=batch.device)
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
        if keepdim:
            out = out.reshape(*batch.batch_size)

        torch.set_grad_enabled(prev)
        return out, out_states

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self
