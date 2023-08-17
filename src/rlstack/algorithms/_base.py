import os
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

import cloudpickle
from tensordict import TensorDict
from torchrl.data import CompositeSpec

from ..data import (
    AlgorithmHparams,
    AlgorithmState,
    CollectStats,
    MemoryStats,
    StepStats,
)
from ..env import Env
from ..optimizer import OptimizerWrapper
from ..policies import GenericPolicyBase
from ..schedulers import EntropyScheduler, LRScheduler

_AlgorithmHparams = TypeVar("_AlgorithmHparams", bound=AlgorithmHparams)
_AlgorithmState = TypeVar("_AlgorithmState", bound=AlgorithmState)
_Policy = TypeVar("_Policy", bound=GenericPolicyBase[Any])


class GenericAlgorithmBase(
    Generic[_AlgorithmHparams, _AlgorithmState, _Policy], metaclass=ABCMeta
):
    #: Environment experience buffer used for aggregating environment
    #: transition data and policy sample data. The same buffer object
    #: is shared whenever using :meth:`GenericAlgorithmBase.collect` Buffer
    #: dimensions are determined by ``num_envs`` and ``horizon`` args.
    buffer: TensorDict

    #: Tensor spec defining the environment experience buffer components
    #: and dimensions. Used for instantiating :attr:`GenericAlgorithmBase.buffer`
    #: at :class:`GenericAlgorithmBase` instantiation and each
    #: :meth:`GenericAlgorithmBase.step` call.
    buffer_spec: CompositeSpec

    #: Entropy scheduler for updating the ``entropy_coeff`` after each
    #: :meth:`Algorithm.step` call based on the number environment transitions
    #: collected and learned on. By default, the entropy scheduler does not
    #: actually update the entropy coefficient. The entropy scheduler only
    #: updates the entropy coefficient if an ``entropy_coeff_schedule`` is
    #: provided.
    entropy_scheduler: EntropyScheduler

    #: Environment used for experience collection within the
    #: :meth:`GenericAlgorithmBase.collect` method. It's ultimately up to the
    #: environment to make learning efficient by parallelizing simulations.
    env: Env

    #: Feedforward PPO hyperparameters that're constant throughout training
    #: and can drastically affect training performance.
    hparams: _AlgorithmHparams

    #: Learning rate scheduler for updating ``optimizer`` learning rate after
    #: each ``step`` call based on the number of environment transitions
    #: collected and learned on. By default, the learning scheduler does not
    #: actually alter the ``optimizer`` learning rate (it actually leaves it
    #: constant). The learning rate scheduler only alters the learning rate
    #: if a ``learning_rate_schedule`` is provided.
    lr_scheduler: LRScheduler

    #: Wrapper around the underlying optimizer for updating the policy's model
    #: that was constructed from ``optimizer_cls`` and ``optimizer_config``.
    #: Handles gradient accumulation and Automatic Mixed Precision (AMP) model
    #: updates. ``optimizer_cls`` defaults to a the Adam optimizer.
    optimizer: OptimizerWrapper

    #: Policy constructed from the ``model_cls``, ``model_config``, and
    #: ``distribution_cls`` kwargs. A default policy is constructed according to
    #: the environment's observation and action specs if these policy args
    #: aren't provided. The policy is what does all the action sampling
    #: within :meth:`GenericAlgorithmBase.collect` and is what is updated within
    #: :meth:`GenericAlgorithmBase.step`.
    policy: _Policy

    #: Algorithm state for determining when to reset the environment, when
    #: the policy can be updated, and for tracking additional algorithm
    #: metrics like time elapsed within a method.
    state: _AlgorithmState

    @abstractmethod
    def collect(
        self,
        *,
        env_config: None | dict[str, Any] = None,
        deterministic: bool = False,
    ) -> CollectStats:
        """Collect environment transitions and policy samples in a buffer.

        This is one of the main :class:`GenericAlgorithmBase` methods. This is
        usually called immediately prior to :meth:`GenericAlgorithmBase.step`
        to collect experiences used for learning.

        The environment is reset immediately prior to collecting
        transitions according to ``horizons_per_env_reset``. If
        the environment isn't reset, then the last observation is used as
        the initial observation.

        This method sets the ``buffered`` flag to enable calling
        of :meth:`Algorithm.step` so it isn't called with dummy data.

        Args:
            env_config: Optional config to pass to the environment's reset
                method. This isn't used if the environment isn't scheduled
                to be reset according to ``horizons_per_env_reset``.
            deterministic: Whether to sample from the policy deterministically.
                This is usally ``False`` during learning and ``True`` during
                evaluation.

        Returns:
            Summary statistics related to the collected experiences and
            policy samples.

        """

    @property
    def horizons_per_env_reset(self) -> int:
        """Number of times :meth:`GenericAlgorithmBase.collect` can be
        called before resetting :attr:`GenericAlgorithmBase.env`. Set this to
        a higher number if you want learning to occur across horizons. Leave
        this as the default ``1`` if it doesn't matter that experiences and
        learning only occurs within one horizon.

        """
        return self.hparams.horizons_per_env_reset

    @abstractmethod
    def memory_stats(self) -> MemoryStats:
        """Return current algorithm memory usage."""

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return algorithm parameters."""

    def save_policy(self, path: str | os.PathLike[str], /) -> None:
        """Save the policy by cloud pickling it to ``path``.

        This method is only defined to expose a common interface between
        different algorithms for saving the underlying policy through
        the trainer interface. This is by no means the only way
        to save a policy and isn't even a recommended way to save
        a policy.

        """
        with open(path, "wb") as f:
            cloudpickle.dump(self.policy, f)

    @abstractmethod
    def step(self) -> StepStats:
        """Take a step with the algorithm, using collected environment
        experiences to update the policy.

        Returns:
            Data associated with the step (losses, loss coefficients, etc.).

        """
