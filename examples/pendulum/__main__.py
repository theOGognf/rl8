import mlflow
import torch

from rl8 import Trainer
from rl8.conditions import HitsUpperBound

from .env import Pendulum

experiment = mlflow.set_experiment("rl8.examples.pendulum")
print(f"Logging runs under MLflow experiment {experiment.experiment_id}")
trainer = Trainer(
    Pendulum,
    horizon=128,
    horizons_per_env_reset=4,
    enable_amp=torch.cuda.is_available(),
    device="auto",
)
trainer.algorithm.validate()
run = mlflow.active_run()
print(f"Logging metrics under MLflow run {run.info.run_id}")
trainer.run(
    steps_per_eval=8,
    stop_conditions=[HitsUpperBound("algorithm/steps", 100)],
)
