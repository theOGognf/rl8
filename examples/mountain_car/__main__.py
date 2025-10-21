import mlflow
import torch

from rl8 import AlgorithmConfig, Trainer
from rl8.conditions import HitsUpperBound

from .env import MountainCar

experiment = mlflow.set_experiment("rl8.examples.mountain_car")
print(f"Logging runs under MLflow experiment {experiment.experiment_id}")
algo = AlgorithmConfig(
    horizon=128,
    horizons_per_env_reset=4,
    enable_amp=torch.cuda.is_available(),
    device="auto",
).build(MountainCar)
trainer = Trainer(algo)
run = mlflow.active_run()
print(f"Logging metrics under MLflow run {run.info.run_id}")
trainer.run(
    stop_conditions=[HitsUpperBound("algorithm/steps", 40)],
)
