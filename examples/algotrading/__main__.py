import argparse

import mlflow
import torch

from rl8 import RecurrentTrainer, Trainer
from rl8.conditions import Plateaus

from .env import AlgoTrading
from .models import AttentiveAlpaca, LazyLemur, MischievousMule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "An example algotrading environment where a policy learns to hold, buy, and"
            " sell an asset. This example serves as a playground for custom,"
            " sequence-based and recurrent models."
        )
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "mlp", "transformer"],
        default="mlp",
        help="Model class type to use.",
    )
    args = parser.parse_args()

    match args.model:
        case "lstm":
            trainer_cls = RecurrentTrainer
            model_cls = LazyLemur
        case "mlp":
            trainer_cls = Trainer
            model_cls = MischievousMule
        case "transformer":
            trainer_cls = Trainer
            model_cls = AttentiveAlpaca

    experiment = mlflow.set_experiment("rl8.examples.algotrading")
    print(f"Logging run under MLflow experiment {experiment.experiment_id}")
    trainer = trainer_cls(
        AlgoTrading,
        model_cls=model_cls,
        enable_amp=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    run = mlflow.active_run()
    print(f"Logging metrics under MLflow run {run.info.run_id}")
    trainer.run(
        steps_per_eval=10,
        stop_conditions=[Plateaus("returns/mean", patience=10, rtol=0.05)],
    )
