from typing import Any

import cloudpickle
import mlflow
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import TensorSpec
from typing_extensions import Self

from rlstack.distributions import Distribution

from .._utils import get_batch_size_from_model_input, td2df
from ..data import DataKeys, Device
from ..models import Model
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
        keepdim: bool = False,
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

    def to(self, device: Device, /) -> Self:
        """Move the policy and its attributes to ``device``."""
        self.model = self.model.to(device)
        return self


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
    define with ``rlstack``. This implementation supports many observation
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
        ... from rlstack import Trainer
        ... from rlstack.env import DiscreteDummyEnv
        ... from rlstack.policies import MLflowPolicyModel
        ... # Create the trainer and step it once for the heck of it.
        ... trainer = Trainer(DiscreteDummyEnv)
        ... trainer.step()
        ... # Create a temporary directory for storing model artifacts
        ... # and the actual MLflow model. This'll get cleaned-up
        ... # once the context ends.
        ... with TemporaryDirectory() as tmp:
        ...     trainer.algorithm.save_policy(f"{tmp}/policy.pkl")
        ...     # This is where you set options specific to your
        ...     # use case. At a bare minimum, the policy's
        ...     # artifact (the policy pickle file) is specified,
        ...     # but you may want to add code files, data files,
        ...     # dependencies/requirements, etc..
        ...     mlflow.pyfunc.save_model(
        ...         f"{tmp}/model",
        ...         python_model=MLflowPolicyModel(),
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
            batch_size=batch_size,
            device=self.policy.device,
        )
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
