from typing import Any

import torch
from tensordict import TensorDict
from typing_extensions import Self

from ..data import DataKeys, Device
from ..distributions import Distribution
from ..models import RecurrentModel
from ..specs import CompositeSpec, TensorSpec


class RecurrentPolicy:
    """The union of a recurrent model and an action distribution.

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
    #: features produced by :attr:`RecurrentPolicy.model`.
    distribution_cls: type[Distribution]

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
            model_cls = model_cls or RecurrentModel.default_model_cls(
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

    def init_states(self, n: int, /) -> TensorDict:
        """Return new recurrent states for the policy's model."""
        return self.model.init_states(n)

    @property
    def observation_spec(self) -> TensorSpec:
        """Return the observation spec used for constructing the model."""
        return self.model.observation_spec

    def sample(
        self,
        batch: TensorDict,
        /,
        states: None | TensorDict = None,
        *,
        deterministic: bool = False,
        inplace: bool = False,
        keepdim: bool = False,
        requires_grad: bool = False,
        return_actions: bool = True,
        return_logp: bool = False,
        return_values: bool = False,
    ) -> tuple[TensorDict, TensorDict]:
        """Use ``batch`` and ``states`` to sample from the policy, sampling
        actions from the model and optionally sampling additional values
        often used for training and analysis.

        Args:
            batch: Batch to feed into the policy's underlying model. Expected
                to be of size ``[B, T, ...]`` where ``B`` is the batch dimension,
                and ``T`` is the time or sequence dimension. ``B`` is typically
                the number of parallel environments being sampled for during
                massively parallel training, and ``T`` is typically the number
                of time steps or observations sampled from the environments.
            states: States to feed into the policy's underlying model. Expected
                to be of size ``[B, T, ...]`` where ``B`` is the batch dimension,
                and ``T`` is the time or sequence dimension. ``B`` is typically
                the number of parallel environments being sampled for during
                massively parallel training, and ``T`` is typically the number
                of time steps or observations sampled from the environments.

            deterministic: Whether to sample from the policy deterministically
                (the actions are always the same for the same inputs) or
                stochastically (there is a randomness to the policy's actions).
            inplace: Whether to store policy outputs in the given ``batch``
                tensordict. Otherwise, create a separate tensordict that
                will only contain policy outputs.
            keepdim: Whether to reshape the output tensordict to have the same
                batch size as the input tensordict batch. If ``False`` (the
                default), the time dimension of the output tensordict will
                be flattened into the first dimension.
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
            A tensordict containing AT LEAST actions sampled from the policy
            and a tensordict containing updated recurrent states. The returned
            recurrent states will only have shape ``[B, ...]`` WITHOUT a
            time dimension ``T`` since only the last recurrent state of the
            series should be returned.

        """
        training = self.model.training
        if deterministic and training:
            self.model.eval()

        # This is the same mechanism within `torch.no_grad`
        # for enabling/disabling gradients.
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(requires_grad)

        B, T = batch.batch_size
        states = self.model.init_states(B).reshape(B, 1) if states is None else states
        features, out_states = self.model(batch, states)

        # Store required outputs and get/store optional outputs.
        out = (
            batch if inplace else TensorDict({}, batch_size=B * T, device=batch.device)
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
        if keepdim:
            out = out.reshape(B, T)

        torch.set_grad_enabled(prev)

        if deterministic and training:
            self.model.train()

        return out, out_states

    @property
    def state_spec(self) -> CompositeSpec:
        """Return the policy's model's state spec for defining recurrent state
        dimensions.

        """
        return self.model.state_spec

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self
