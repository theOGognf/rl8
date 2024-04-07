"""High-level training interfaces."""

from typing import Any, Literal, TypedDict

import torch.optim as optim
from typing_extensions import Unpack

from ..algorithms import Algorithm
from ..data import Device
from ..distributions import Distribution
from ..env import EnvFactory
from ..models import Model, ModelFactory
from ..schedulers import ScheduleKind
from ._base import GenericTrainerBase


class AlgorithmConfig(TypedDict, total=False):
    """Just a container for algorithm kwargs to reduce repitition.
    See :class:`Algorithm` for more details.

    """

    env_config: None | dict[str, Any]
    model: None | Model
    model_cls: None | ModelFactory
    model_config: None | dict[str, Any]
    distribution_cls: None | type[Distribution]
    horizon: int
    horizons_per_env_reset: int
    num_envs: int
    optimizer_cls: type[optim.Optimizer]
    optimizer_config: None | dict[str, Any]
    accumulate_grads: bool
    enable_amp: bool
    lr_schedule: None | list[tuple[int, float]]
    lr_schedule_kind: ScheduleKind
    entropy_coeff: float
    entropy_coeff_schedule: None | list[tuple[int, float]]
    entropy_coeff_schedule_kind: ScheduleKind
    gae_lambda: float
    gamma: float
    sgd_minibatch_size: None | int
    num_sgd_iters: int
    shuffle_minibatches: bool
    clip_param: float
    vf_clip_param: float
    dual_clip_param: None | float
    vf_coeff: float
    target_kl_div: None | float
    max_grad_norm: float
    normalize_advantages: bool
    normalize_rewards: bool
    device: Device | Literal["auto"]


class Trainer(GenericTrainerBase[Algorithm]):
    """Higher-level training interface that interops with other tools for
    tracking and saving experiments (i.e., MLflow).

    This is the preferred training interface when training feedforward
    (i.e., non-recurrent) policies in most cases.

    Args:
        env_cls: Highly parallelized environment for sampling experiences.
            Instantiated with ``env_config``. Will be stepped for ``horizon``
            each :meth:`Algorithm.collect` call.
        **algorithm_config: See :class:`Algorithm`.

    """

    def __init__(
        self, env_cls: EnvFactory, /, **algorithm_config: Unpack[AlgorithmConfig]
    ) -> None:
        super().__init__(Algorithm(env_cls, **algorithm_config))
