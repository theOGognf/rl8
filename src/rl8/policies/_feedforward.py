import os
from typing import Any

import cloudpickle
import mlflow
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import TensorSpec

from rl8.distributions import Distribution

from .._utils import get_batch_size_from_model_input, td2df
from ..data import DataKeys, Device
from ..models import Model, ModelFactory
from ..views import ViewKind
from ._base import GenericPolicyBase


class Policy(GenericPolicyBase[Model]):
    """The union of a feedforward model and an action distribution.

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
        model: None | Model = None,
        model_cls: None | ModelFactory = None,
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
            A tensordict containing AT LEAST actions sampled from the policy of
            batch size ``[B * T, ...]`` where ``B`` is the input's batch dimension,
            and ``T`` is the time or sequence dimension.

        """
        if DataKeys.VIEWS in batch.keys():
            in_batch = batch[DataKeys.VIEWS]
        else:
            in_batch = self.model.apply_view_requirements(batch, kind=kind)

        # Should be in eval mode when `deterministic` is `True`.
        # That is, `training` should be the opposite of `deterministic`.
        training = self.model.training
        if deterministic == training:
            self.model.train(not training)

        # This is the same mechanism within `torch.no_grad`
        # for enabling/disabling gradients.
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(requires_grad)

        features = self.model(in_batch)

        # Store required outputs and get/store optional outputs.
        out = (
            batch.reshape(-1)
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

        if deterministic == training:
            self.model.train(training)

        return out

    def save(self, path: str | os.PathLike[str], /) -> "MLflowPolicyModel":
        """Save the policy by cloud pickling it to ``path`` and returning
        the interface used for deploying it with MLflow.

        This method is only defined to expose a common interface between
        different algorithms. This is by no means the only way
        to save a policy and isn't even the recommended way to save
        a policy.

        """
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)
        return MLflowPolicyModel()


class MLflowPolicyModel(mlflow.pyfunc.PythonModel):
    """A MLflow Python model implementation of a feedforward policy.

    This is by no means the only way to define a MLflow interface for
    a feedforward policy, nor is it the recommended way to deploy
    or serve your trained policy with MLflow. This is simply a minimal
    and generic implementation of a MLflow Python model for feedforward
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
        ... from rl8 import Trainer
        ... from rl8.env import DiscreteDummyEnv
        ... # Create the trainer and step it once for the heck of it.
        ... trainer = Trainer(DiscreteDummyEnv)
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
        self.policy: Policy = cloudpickle.load(open(context.artifacts["policy"], "rb"))

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: dict[str, Any],
    ) -> pd.DataFrame:
        """Sample from the underlying policy using ``model_input`` as input.

        Args:
            context: Python model context that's unused for this implementation.
            model_input: Policy model input (or observation). The observation
                space is expected to be a 1D vector or a composite spec that
                maps strings to tensor specs; the policy model is expected to
                ingest a tensordict and handle all the input preprocessing
                (such as tensor concatenation) on its own. The model input
                (or observation) is expected to match the policy model's
                observation space within an ``"obs"`` key and is expected to
                be of shape ``[B, T, ...]`` for each tensor within the observation
                where ``B`` is the batch dimension, and ``T`` is the time or
                sequence dimension. The underlying policy will handle reshaping
                of the model input for batch inference and the policy's outputs
                will be of shape ``[B * T, ...]`` such that the batch and time
                dimensions are flattened into the first dimension. Thus,
                the index of the resulting output dataframe from this method
                will correspond to indicies of the flattened first dimension.

        Returns:
            A dataframe with ``B * T`` rows containing sampled actions, log
            probabilities of sampling those actions, and value estimates.
            ``B`` is the model input's batch dimension, and ``T`` is the model
            input's time or sequence dimension.

        """
        obs = model_input[DataKeys.OBS]
        batch_size = get_batch_size_from_model_input(obs)
        batch = TensorDict(
            {DataKeys.OBS: self.policy.observation_spec.encode(obs)},
            batch_size=[],
            device=self.policy.device,
        )
        batch = batch.reshape(*batch_size)
        batch = self.policy.sample(
            batch,
            kind="all",
            deterministic=True,
            inplace=False,
            requires_grad=False,
            return_actions=True,
            return_logp=True,
            return_values=True,
            return_views=False,
        )
        batch = batch.select(
            DataKeys.ACTIONS, DataKeys.LOGP, DataKeys.VALUES, inplace=True
        )
        return td2df(batch)
