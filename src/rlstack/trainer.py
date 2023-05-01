from typing import Any

import mlflow

from .algorithm import Algorithm
from .conditions import Condition
from .data import TrainStats
from .env import Env


class Trainer:
    def __init__(
        self,
        env_cls: type[Env],
        /,
        *,
        algorithm_config: None | dict[str, Any] = None,
        stop_conditions: None | list[Condition] = None,
    ) -> None:
        algorithm_config = algorithm_config or {}
        self.algorithm = Algorithm(env_cls, **algorithm_config)
        self.stop_conditions = stop_conditions or []
        mlflow.log_params(self.algorithm.params)

    def run(self, *, env_config: None | dict[str, Any] = None) -> TrainStats:
        train_stats = self.train(env_config=env_config)
        while not any([condition(train_stats) for condition in self.stop_conditions]):
            train_stats = self.train(env_config=env_config)
        return train_stats

    def train(self, *, env_config: None | dict[str, Any] = None) -> TrainStats:
        train_stats = {
            **self.algorithm.collect(env_config=env_config),
            **self.algorithm.step(),
        }
        mlflow.log_metrics(train_stats, step=train_stats["counting/total_steps"])
        return train_stats  # type: ignore[return-value]
