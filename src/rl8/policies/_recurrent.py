import os
from typing import Any

import cloudpickle
import mlflow
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, TensorSpec

from .._utils import get_batch_size_from_model_input, td2df
from ..data import DataKeys, Device
from ..distributions import Distribution
from ..models import RecurrentModel, RecurrentModelFactory
from ._base import GenericPolicyBase


class RecurrentPolicy(GenericPolicyBase[RecurrentModel]):  # type: ignore[type-var]
    """The union of a recurrent model and an action distribution.

    Args:
        observation_spec: Spec defining observations from the environment
            and inputs to the model's forward pass.
        action_spec: Spec defining the action distribution's outputs
            and the inputs to the environment.
        model: Model instance to use. Mutually exclusive with ``model_cls``.
        model_cls: Model class or class factory to use.
        model_config: Model class args.
        distribution_cls: Action distribution class.

    """

    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        *,
        model: None | RecurrentModel = None,
        model_cls: None | RecurrentModelFactory = None,
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

    def init_states(self, n: int, /) -> TensorDict:
        """Return new recurrent states for the policy's model."""
        return self.model.init_states(n)

    def sample(
        self,
        batch: TensorDict,
        /,
        states: None | TensorDict = None,
        *,
        deterministic: bool = False,
        inplace: bool = False,
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
            series should be returned. Other returned values will have batch
            size ``[B * T, ...]`` where ``B`` is the input's batch dimension,
            and ``T`` is the time or sequence dimension.

        """
        # Should be in eval mode when `deterministic` is `True`.
        # That is, `training` should be the opposite of `deterministic`.
        training = self.model.training
        if deterministic == training:
            self.model.train(not training)

        # This is the same mechanism within `torch.no_grad`
        # for enabling/disabling gradients.
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(requires_grad)

        B, T = batch.batch_size
        states = self.model.init_states(B).reshape(B, 1) if states is None else states
        features, out_states = self.model(batch, states)

        # Store required outputs and get/store optional outputs.
        out = (
            batch.reshape(B * T)
            if inplace
            else TensorDict({}, batch_size=B * T, device=batch.device)
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

        torch.set_grad_enabled(prev)

        if deterministic == training:
            self.model.train(training)

        return out, out_states

    def save(self, path: str | os.PathLike[str], /) -> "MLflowRecurrentPolicyModel":
        """Save the policy by cloud pickling it to ``path`` and returning
        the interface used for deploying it with MLflow.

        This method is only defined to expose a common interface between
        different algorithms. This is by no means the only way
        to save a policy and isn't even the recommended way to save
        a policy.

        """
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)
        return MLflowRecurrentPolicyModel()

    @property
    def state_spec(self) -> CompositeSpec:
        """Return the policy's model's state spec for defining recurrent state
        dimensions.

        """
        return self.model.state_spec


class MLflowRecurrentPolicyModel(mlflow.pyfunc.PythonModel):
    """A MLflow Python model implementation of a recurrent policy.

    This is by no means the only way to define a MLflow interface for
    a recurrent policy, nor is it the recommended way to deploy
    or serve your trained policy with MLflow. This is simply a minimal
    and generic implementation of a MLflow Python model for recurrent
    policies that serves as a convenience. The majority of policy deployment
    use cases will probably be satisified with this minimal implementation.
    Use cases not covered by this implementation are encouraged to write
    their own implementation that fits their needs as this implementation
    will likely not see further development beyond bugfixes.

    On top of this implementation being minimal and in "maintenance mode",
    it doesn't support all the many kinds of policy models one could
    define with ``rl8``. This implementation supports many observation
    spaces, but this implementation does not support all action spaces.
    Action spaces are limited to (flattened) 1D spaces; more than 1D is
    possible, but it's likely it will experience inconsistent behavior
    when storing actions in the output dataframe.

    Examples:

        A minimal example of training a policy, saving it with MLflow,
        and then reloading it for inference using this interface.

        >>> from tempfile import TemporaryDirectory
        ...
        ... import mlflow
        ...
        ... from rl8 import RecurrentTrainer
        ... from rl8.env import DiscreteDummyEnv
        ... # Create the trainer and step it once for the heck of it.
        ... trainer = RecurrentTrainer(DiscreteDummyEnv)
        ... trainer.step()
        ... # Create a temporary directory for storing model artifacts
        ... # and the actual MLflow model. This'll get cleaned-up
        ... # once the context ends.
        ... with TemporaryDirectory() as tmp:
        ...     # This is where you set options specific to your
        ...     # use case. At a bare minimum, the policy's
        ...     # artifact (the policy pickle file) is specified,
        ...     # but you may want to add code files, data files,
        ...     # dependencies/requirements, etc..
        ...     mlflow.pyfunc.save_model(
        ...         f"{tmp}/model",
        ...         python_model=trainer.algorithm.policy.save(f"{tmp}/policy.pkl"),
        ...         artifacts={"policy": f"{tmp}/policy.pkl"},
        ...     )
        ...     model = mlflow.pyfunc.load_model(f"{tmp}/model")
        ...     # We cheat here a bit and use the environment's spec
        ...     # to generate a valid input example. These are usually
        ...     # constructed by some other service.
        ...     obs = DiscreteDummyEnv(1).observation_spec.rand([1, 1]).cpu().numpy()
        ...     model.predict({"obs": obs})  # doctest: +SKIP

    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Loads the saved policy on model instantiation."""
        self.policy: RecurrentPolicy = cloudpickle.load(
            open(context.artifacts["policy"], "rb")
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: dict[str, Any],
    ) -> list[pd.DataFrame]:
        """Sample from the underlying policy using ``model_input`` as input.

        Args:
            context: Python model context that's unused for this implementation.
            model_input: Policy model input (or observation). The observation
                space is expected to be a composite spec that maps strings to
                tensor specs; the policy model is expected to ingest a
                tensordict and handle all the input preprocessing
                (such as tensor concatenation) on its own. The model input
                (or observation) is expected to match the policy model's
                observation space and should contain the recurrent model's
                recurrent state (unless a new recurrent state is
                to be instantiated). The model inputs are expected to be of shape
                ``[B, T, ...]`` for each tensor within the observation where
                ``B`` is the batch dimension, and ``T`` is the time or sequence
                dimension, while the model recurrent states are expected to be
                of shape ``[B, 1, ...]``. The underlying policy will handle
                reshaping of the model input for batch inference and the policy's
                outputs will be of shape ``[B * T, ...]`` such that the batch and
                time dimensions are flattened into the first dimension. Thus,
                the index of the resulting output dataframe from this method
                will correspond to indicies of the flattened first dimension.
                The output dataframe will also contain the updated recurrent
                states for just the final timestep. These recurrent states are
                repeated along the time dimension to allow storing of recurrent
                states within the same dataframe as the model outputs.

        Returns:
            Two dataframes: the first with ``B * T`` rows containing sampled
            actions, log probabilities of sampling those actions, and value
            estimates; the second with ``B`` rows containing updated recurrent
            model states. ``B`` is the model input's batch dimension, and ``T``
            is the model input's time or sequence dimension.

        """
        obs = model_input[DataKeys.OBS]
        batch_size = get_batch_size_from_model_input(obs)
        batch = TensorDict(
            {DataKeys.OBS: self.policy.observation_spec.encode(obs)},
            batch_size=[],
            device=self.policy.device,
        )
        batch = batch.reshape(*batch_size)
        if DataKeys.STATES in model_input:
            states = TensorDict(
                self.policy.model.state_spec.encode(model_input[DataKeys.STATES]),
                batch_size=[],
                device=self.policy.device,
            )
            states = states.reshape(batch_size[0], 1)
        else:
            states = None
        batch, states = self.policy.sample(
            batch,
            states,
            deterministic=True,
            inplace=False,
            requires_grad=False,
            return_actions=True,
            return_logp=True,
            return_values=True,
        )
        batch = batch.select(
            DataKeys.ACTIONS, DataKeys.LOGP, DataKeys.VALUES, inplace=True
        )
        return [td2df(batch), td2df(states)]
