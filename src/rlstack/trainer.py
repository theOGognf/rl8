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

    def run(self) -> TrainStats:
        mlflow.start_run()
        train_stats: TrainStats = {}
        while not any(
            [condition.eval(train_stats) for condition in self.stop_conditions]
        ):
            train_stats = self.train()
        mlflow.end_run()
        return train_stats

    def train(self) -> TrainStats:
        return {**self.algorithm.collect(), **self.algorithm.step()}  # type: ignore[misc]
