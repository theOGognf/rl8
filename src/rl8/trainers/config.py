"""Configuration for the high-level training interfaces."""

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch.optim as optim
import yaml

from ..data import Device
from ..distributions import Distribution
from ..env import EnvFactory
from ..models import ModelFactory, RecurrentModelFactory
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
            horizon: 8
            gamma: 1

        The following will instantiate a :class:`TrainConfig` from the
        file, instantiate a trainer, and then train indefinitely.

        >>> from rl8 import TrainConfig
        >>> TrainConfig.from_file("./config.yaml").build().run()

    """

    env_cls: EnvFactory
    env_config: None | dict[str, Any] = None
    model_cls: None | RecurrentModelFactory | ModelFactory = None
    model_config: None | dict[str, Any] = None
    distribution_cls: None | type[Distribution] = None
    horizon: None | int = None
    horizons_per_env_reset: None | int = None
    num_envs: None | int = None
    seq_len: None | int = None
    seqs_per_state_reset: None | int = None
    optimizer_cls: None | type[optim.Optimizer] = None
    optimizer_config: None | dict[str, Any] = None
    accumulate_grads: None | bool = None
    enable_amp: None | bool = None
    entropy_coeff: None | float = None
    gae_lambda: None | float = None
    gamma: None | float = None
    sgd_minibatch_size: None | int = None
    num_sgd_iters: None | int = None
    shuffle_minibatches: None | bool = None
    clip_param: None | float = None
    vf_clip_param: None | float = None
    dual_clip_param: None | float = None
    vf_coeff: None | float = None
    target_kl_div: None | float = None
    max_grad_norm: None | float = None
    normalize_advantages: None | bool = None
    normalize_rewards: None | bool = None
    device: None | Device | Literal["auto"] = None
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
            >>> from rl8 import DiscreteDummyEnv, TrainConfig
            >>> trainer = TrainConfig(DiscreteDummyEnv).build()

        """
        config = asdict(self)
        recurrent = config.pop("recurrent")
        filtered_config = {k: v for k, v in config.items() if v is not None}
        env_cls = filtered_config.pop("env_cls")
        return (
            RecurrentTrainer(env_cls, **filtered_config)
            if recurrent
            else Trainer(env_cls, **filtered_config)
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

        if "env_cls" in data:
            env_cls = _import(data.pop("env_cls"))
        else:
            raise RuntimeError(f"{cls.__name__} config {path} must contain `env_cls`")

        for k in ("model_cls", "distribution_cls", "optimizer_cls"):
            if k in data:
                data[k] = _import(data[k])

        return cls(env_cls, **data)
