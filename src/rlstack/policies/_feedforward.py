from typing import Any

import torch
from tensordict import TensorDict
from typing_extensions import Self

from rlstack.distributions import Distribution

from ..data import DataKeys, Device
from ..models import Model
from ..specs import TensorSpec
from ..views import ViewKind


class Policy:
    """The union of a feedforward model and an action distribution.

    Args:
        observation_spec: Spec defining observations from the environment
            and inputs to the model's forward pass.
        action_spec: Spec defining the action distribution's outputs
            and the inputs to the environment.
        model: Model instance to use. Mutually exclusive with ``model_cls``.
        model_cls: Model class to use.
        model_config: Model class args.
        distribution_cls: Action distribution class.

    """

    #: Underlying policy action distribution that's parameterized by
    #: features produced by :attr:`Policy.model`.
    distribution_cls: type[Distribution]

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
        distribution_cls: None | type[Distribution] = None,
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
        self.distribution_cls = distribution_cls or Distribution.default_dist_cls(
            action_spec
        )

    @property
    def action_spec(self) -> TensorSpec:
        """Return the action spec used for constructing the model."""
        return self.model.action_spec

    @property
    def device(self) -> Device:
        """Return the device the policy's model is on."""
        return next(self.model.parameters()).device

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
                tensordict. Otherwise, create a separate tensordict that
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
            A tensordict containing AT LEAST actions sampled from the policy.

        """
        if DataKeys.VIEWS in batch.keys():
            in_batch = batch[DataKeys.VIEWS]
        else:
            in_batch = self.model.apply_view_requirements(batch, kind=kind)

        training = self.model.training
        if deterministic and training:
            self.model.eval()

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
            dist = self.distribution_cls(features, self.model)
            actions = dist.deterministic_sample() if deterministic else dist.sample()
            out[DataKeys.ACTIONS] = actions
            if return_logp:
                out[DataKeys.LOGP] = dist.logp(actions)
        if return_values:
            out[DataKeys.VALUES] = self.model.value_function()
        if return_views:
            out[DataKeys.VIEWS] = in_batch

        torch.set_grad_enabled(prev)

        if deterministic and training:
            self.model.train()

        return out

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self
