"""High-level training interfaces."""

from collections import defaultdict
from typing import Any, Protocol

import mlflow

from ._utils import reduce_stats
from .algorithms import Algorithm
from .conditions import Condition
from .data import (
    AlgorithmHparams,
    CollectStats,
    EvalCollectStats,
    MemoryStats,
    StepStats,
    TrainerState,
    TrainStats,
)
from .env import Env


class AlgorithmProtocol(Protocol):
    """Protocol for algorithms used by the trainer for training policies."""

    hparams: AlgorithmHparams

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

    def memory_stats(self) -> MemoryStats:
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

    """

    #: Underlying PPO algorithm, including the environment, model,
    #: action distribution, and hyperparameters.
    algorithm: AlgorithmProtocol

    #: Trainer state used for tracking a handful of running totals
    #: necessary for logging metrics, determining when a policy
    #: can be evaluated, etc..
    state: TrainerState

    def __init__(
        self,
        env_cls: type[Env],
        /,
        *,
        algorithm_cls: None | type[AlgorithmProtocol] = None,
        algorithm_config: None | dict[str, Any] = None,
    ) -> None:
        algorithm_config = algorithm_config or {}
        algorithm_cls = algorithm_cls or Algorithm
        self.algorithm = algorithm_cls(env_cls, **algorithm_config)
        self.state = {
            "algorithm/collects": 0,
            "algorithm/steps": 0,
            "env/steps": 0,
        }
        mlflow.log_params(self.algorithm.params)

    def eval(
        self, *, env_config: None | dict[str, Any] = None, deterministic: bool = True
    ) -> EvalCollectStats:
        """Run a single evaluation step, collecting environment transitions
        for several horizons with potentially different environment configs.

        Args:
            env_config: Environment config override. Useful for evaluating a
                policy's generalizability by setting the environment config
                to something different from the environment config during
                training.
            deterministic: Whether to sample from the policy deterministically.
                This is usally ``False`` during learning and ``True`` during
                evaluation.

        Returns:
            Eval stats from the collection buffer.

        """
        if (
            self.state["algorithm/collects"]
            % self.algorithm.hparams.horizons_per_env_reset
        ):
            raise RuntimeError(
                f"{self.eval.__qualname__} can only be called every"
                " `horizons_per_env_reset`. This is necessary because algorithms share"
                " the same buffer when collecting experiences for training and for"
                " evaluation."
            )
        stats = defaultdict(list)
        for _ in range(self.algorithm.hparams.horizons_per_env_reset):
            for k, v in self.algorithm.collect(
                env_config=env_config, deterministic=deterministic
            ).items():
                stats[k].append(v)
            self.state["algorithm/collects"] += 1
        eval_stats = {f"eval/{k}": v for k, v in reduce_stats(stats).items()}  # type: ignore[arg-type]
        mlflow.log_metrics(eval_stats, step=self.state["env/steps"])
        return eval_stats  # type: ignore[return-value]

    def run(
        self,
        *,
        env_config: None | dict[str, Any] = None,
        eval_env_config: None | dict[str, Any] = None,
        steps_per_eval: None | int = None,
        stop_conditions: None | list[Condition] = None,
    ) -> TrainStats:
        """Run the trainer and underlying algorithm until at least of of the
        ``stop_conditions`` is satisfied.

        This method runs indefinitely unless at least one stop condition is
        provided.

        Args:
            env_config: Environment config override. Useful for scheduling
                domain randomization.
            eval_env_config: Environment config override during evaluations.
                Defaults to the config provided by ``env_config`` if not
                provided. Useful for evaluating a policy's generalizability.
            steps_per_eval: Number of :meth:`Trainer.step` calls before calling
                :meth:`Trainer.eval`.
            stop_conditions: Conditions evaluated each iteration that determines
                whether to stop training. Only one condition needs to evaluate
                as ``True`` for training to stop. Training will continue
                indefinitely unless a stop condition returns ``True``.

        Returns:
            The most recent train stats when the training is stopped due
            to a stop condition being satisfied.

        """
        if (
            steps_per_eval
            and steps_per_eval % self.algorithm.hparams.horizons_per_env_reset
        ):
            raise ValueError(
                f"{self.eval.__qualname__} can only be called every"
                " `horizons_per_env_reset`. This is necessary because algorithms share"
                " the same buffer for collecting experiences during training and for"
                " evaluation. Set `steps_per_eval` to a factor of"
                " `horizons_per_env_reset to avoid this error."
            )
        eval_env_config = eval_env_config or env_config
        stop_conditions = stop_conditions or []
        train_stats = self.step(env_config=env_config)
        while not any([condition(train_stats) for condition in stop_conditions]):
            if steps_per_eval and not (self.state["algorithm/steps"] % steps_per_eval):
                self.eval(env_config=eval_env_config)
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
        memory_stats = self.algorithm.memory_stats()
        collect_stats = self.algorithm.collect(env_config=env_config)
        step_stats = self.algorithm.step()
        train_stats = {
            **memory_stats,
            **collect_stats,
            **step_stats,
        }
        self.state["algorithm/collects"] += 1
        self.state["algorithm/steps"] += 1
        self.state["env/steps"] += collect_stats["env/steps"]
        mlflow.log_metrics(train_stats, step=self.state["env/steps"])
        return train_stats  # type: ignore[return-value]
