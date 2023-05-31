import argparse

import mlflow
import torch

from rlstack import Algorithm, RecurrentAlgorithm, Trainer
from rlstack.conditions import Plateaus

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
            algorithm_cls = RecurrentAlgorithm
            model_cls = LazyLemur
        case "mlp":
            algorithm_cls = Algorithm
            model_cls = MischievousMule
        case "transformer":
            algorithm_cls = Algorithm
            model_cls = AttentiveAlpaca

    experiment = mlflow.set_experiment("rlstack.examples.algotrading")
    print(f"Logging run under MLFlow experiment {experiment.experiment_id}")
    trainer = Trainer(
        AlgoTrading,
        algorithm_cls=algorithm_cls,
        algorithm_config={
            "model_cls": model_cls,
            "enable_amp": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        stop_conditions=[Plateaus("returns/mean", patience=10, rtol=0.05)],
    )
    run = mlflow.active_run()
    print(f"Logging metrics under MLFlow run {run.info.run_id}")
    trainer.run()
