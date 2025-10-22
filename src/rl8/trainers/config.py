"""Configuration for the high-level training interfaces."""

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..algorithms import AlgorithmConfig, RecurrentAlgorithmConfig
from ..env import EnvFactory
from ._feedforward import Trainer
from ._recurrent import RecurrentTrainer


def _import(name: str) -> Any:
    try:
        components = name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
    except (AttributeError, ModuleNotFoundError, ValueError) as e:
        raise ImportError(f"Could not dynamically import {name}.") from e
    return mod


@dataclass
class TrainConfig:
    """A helper for instantiating a trainer based on a config.

    It's common to run training experiments based on some config.
    This class is introduced to reduce the need for creating some
    custom trainer config parser. This class doesn't support all
    trainer/algorithm options/fields, but supports enough to cover
    the majority of use cases.

    It's intended for this class to be instantiated from a JSON or
    YAML file, and then for the instance to build a trainer directly
    afterwards.

    Examples:
        Assume there's some YAML config at ``./config.yaml`` with the
        following contents:

        .. code-block:: yaml

            env_cls: rl8.env.DiscreteDummyEnv
            algorithm_config:
                horizon: 8
                gamma: 1

        The following will instantiate a :class:`TrainConfig` from the
        file, instantiate a trainer, and then train indefinitely.

        >>> from rl8 import TrainConfig
        >>> TrainConfig.from_file("./config.yaml").build().run()

    """

    #: Environment class to instantiate an algorithm with.
    env_cls: EnvFactory

    #: Algorithm hyperparameters and config to build an algorithm with.
    algorithm_config: dict[str, Any] = field(default_factory=dict)

    #: Whether to instantiate a recurrent variant of the algorithm.
    recurrent: bool = False

    def build(self) -> Trainer | RecurrentTrainer:
        """Instantiate a trainer from the train config.

        Null fields are removed from the train config before being unpacked
        into the trainer's constructor (so the default values on the trainer
        are used to instantiate the trainer). The trainer type (i.e., recurrent
        or feedforward) is specified by the ``recurrent` attribute.

        Returns:
            A trainer based on the train config values.

        Examples:
            >>> from rl8 import TrainConfig
            >>> from rl8.env import DiscreteDummyEnv
            >>> trainer = TrainConfig(DiscreteDummyEnv).build()

        """
        return (
            RecurrentTrainer(
                RecurrentAlgorithmConfig(**self.algorithm_config).build(self.env_cls)
            )
            if self.recurrent
            else Trainer(AlgorithmConfig(**self.algorithm_config).build(self.env_cls))
        )

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "TrainConfig":
        """Instantiate a :class:`TrainConfig` from a JSON or YAML file.

        The JSON or YAML file should have fields with the same type
        as the dataclass fields except for:

            - "env_cls"
            - "model_cls"
            - "distribution_cls"
            - "optimizer_cls"

        These fields should be fully qualified paths to their definitions.
        As an example, if one were to use a custom package ``my_package``
        with submodule ``envs`` and environment class ``MyEnv``, they would
        set ``"env_cls"`` to ``"my_package.envs.MyEnv"``.

        Definitions specified in these fields will be dynamically imported
        from their respective packages and modules. A current limitation is
        these field specifications must point to an installed package and
        can't be from relative file locations (e.g., something like
        ``"..my_package.envs.MyEnv"`` will not work).

        Args:
            path: Pathlike to the JSON or YAML file to read.

        Returns:
            A train config based on the given file.

        """
        p = pathlib.Path(path)
        with open(p, "r") as f:
            match p.suffix:
                case ".json":
                    data = json.load(f)
                case ".yaml":
                    data = yaml.safe_load(f)
                case _:
                    raise ValueError("Config must be a JSON or YAML file")

        if "env_cls" in data:
            data["env_cls"] = _import(data["env_cls"])
        else:
            raise RuntimeError(f"{cls.__name__} config {path} must contain `env_cls`")

        if "algorithm_config" in data:
            for k in ("model_cls", "distribution_cls", "optimizer_cls"):
                if k in data["algorithm_config"]:
                    data["algorithm_config"][k] = _import(data["algorithm_config"][k])

        return cls(**data)
