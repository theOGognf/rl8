"""High-level training interfaces."""

from typing import Any, Protocol

import mlflow

from .algorithms import Algorithm
from .conditions import Condition
from .data import CollectStats, StepStats, TrainStats
from .env import Env


class AlgorithmProtocol(Protocol):
    """Protocol for algorithms used by the trainer for training policies."""

    def __init__(
        self,
        env_cls: type[Env],
        /,
        **kwargs: Any,
    ) -> None:
        ...

    def collect(
        self, *, env_config: None | dict[str, Any] = None, deterministic: bool = False
    ) -> CollectStats:
        ...

    @property
    def params(self) -> dict[str, Any]:
        ...

    def step(self) -> StepStats:
        ...


class Trainer:
    """Higher-level training interface that interops with other tools for
    tracking and saving experiments (i.e., MLFlow).

    This is the preferred training interface for most use cases.

    Args:
        env_cls: Environment to train on.
        algorithm_cls: Algorithm class to use that ``algorithm_config`` is
            unpacked into.
        algorithm_config: Algorithm hyperparameters used to instantiate
            the algorithm. Custom models, model configs, and distributions
            are provided here.
        stop_conditions: Conditions evaluated each iteration within
            :meth:`Trainer.run` that determines whether to stop training.
            Only one condition needs to evaluate as ``True`` for training to
            stop. Training will continue indefinitely unless a stop
            condition returns ``True``.

    """

    #: Underlying PPO algorithm, including the environment, model,
    #: action distribution, and hyperparameters.
    algorithm: AlgorithmProtocol

    #: Conditions evaluated each iteration within :meth:`Trainer.run`
    #: that determines whether to stop training. Only one condition
    #: needs to evaluate as ``True`` for training to stop. Training
    #: will continue indefinitely unless a stop condition returns
    #: ``True``.
    stop_conditions: list[Condition]

    def __init__(
        self,
        env_cls: type[Env],
        /,
        *,
        algorithm_cls: None | type[AlgorithmProtocol] = None,
        algorithm_config: None | dict[str, Any] = None,
        stop_conditions: None | list[Condition] = None,
    ) -> None:
        algorithm_config = algorithm_config or {}
        algorithm_cls = algorithm_cls or Algorithm
        self.algorithm = algorithm_cls(env_cls, **algorithm_config)
        self.stop_conditions = stop_conditions or []
        mlflow.log_params(self.algorithm.params)

    def run(self, *, env_config: None | dict[str, Any] = None) -> TrainStats:
        """Run the trainer and underlying algorithm until at least of of the
        :attr:`Trainer.stop_conditions` is satisfied.

        This method runs indefinitely unless at least one stop condition is
        provided.

        Args:
            env_config: Environment config override. Useful for scheduling
                domain randomization.

        Returns:
            The most recent train stats when the training is stopped due
            to a stop condition being satisfied.

        """
        train_stats = self.step(env_config=env_config)
        while not any([condition(train_stats) for condition in self.stop_conditions]):
            train_stats = self.step(env_config=env_config)
        return train_stats

    def step(self, *, env_config: None | dict[str, Any] = None) -> TrainStats:
        """Run a single training step, collecting environment transitions
        and updating the policy with those transitions.

        Args:
            env_config: Environment config override. Useful for scheduling
                domain randomization.

        Returns:
            Train stats from the policy update.

        """
        train_stats = {
            **self.algorithm.collect(env_config=env_config),
            **self.algorithm.step(),
        }
        mlflow.log_metrics(train_stats, step=train_stats["counting/total_steps"])
        return train_stats  # type: ignore[return-value]
