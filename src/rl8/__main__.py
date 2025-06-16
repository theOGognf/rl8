"""Main CLI entry points."""

import argparse
import pathlib
from typing import Literal

import mlflow

from .conditions import HitsUpperBound
from .trainers import TrainConfig


def _fullname(o):  # type: ignore[no-untyped-def]
    # Credit: https://stackoverflow.com/a/13653312
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__
    return f"{module}.{o.__name__}"


def main() -> Literal[0]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Training command to train a policy using the trainer interface. Provides a"
            " common training workflow that satisfies most use cases. Includes training"
            " from a config file, MLflow experiment/run setup, training validation,"
            " policy saving, and more."
        ),
    )
    train_parser.add_argument(
        "-f",
        "--file",
        type=pathlib.Path,
        help="Train config file to build the trainer from.",
    )
    train_parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "MLflow experiment name to organize runs under. Defaults to the"
            " environment's fully qualified name."
        ),
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help=(
            "Max trainer steps (each trainer step = number of environments * horizon)"
            " before stopping."
        ),
    )
    train_parser.add_argument(
        "--save", default=None, help="Directory to save the trained policy to."
    )
    train_parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=None,
        help="Number of trainer steps for each trainer eval.",
    )

    args = parser.parse_args()

    match args.command:
        case "train":
            config = TrainConfig.from_file(args.file)
            experiment_name = args.experiment_name or _fullname(config.env_cls)
            experiment = mlflow.set_experiment(experiment_name)
            print(f"Logging runs under MLflow experiment {experiment.name}")
            trainer = config.build()
            trainer.algorithm.validate()
            run = mlflow.active_run()
            if run is not None:
                print(f"Logging metrics under MLflow run {run.info.run_name}")
                trainer.run(
                    steps_per_eval=args.steps_per_eval,
                    stop_conditions=[HitsUpperBound("algorithm/steps", args.max_steps)],
                )
                if args.save:
                    pathlib.Path(args.save).mkdir(exist_ok=True)
                    trainer.algorithm.policy.model.eval()
                    trainer.algorithm.policy.to("cpu")
                    mlflow.pyfunc.save_model(
                        f"{args.save}/model",
                        python_model=trainer.algorithm.policy.save(
                            f"{args.save}/policy.pkl"
                        ),
                        artifacts={"policy": f"{args.save}/policy.pkl"},
                        metadata={
                            "experiment_name": experiment.name,
                            "run_name": run.info.run_name,
                        },
                    )
                mlflow.end_run()
    return 0


if __name__ == "__main__":
    SystemExit(main())
