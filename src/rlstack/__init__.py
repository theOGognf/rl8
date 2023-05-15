"""Top-level package interface."""

from importlib.metadata import PackageNotFoundError, version

from .algorithms import Algorithm, RecurrentAlgorithm
from .distributions import Categorical, Distribution, Normal, SquashedNormal
from .env import Env, GenericEnv
from .models import (
    DefaultContinuousModel,
    DefaultContinuousRecurrentModel,
    DefaultDiscreteModel,
    DefaultDiscreteRecurrentModel,
    GenericModel,
    GenericRecurrentModel,
    Model,
    RecurrentModel,
)
from .policies import Policy, RecurrentPolicy
from .trainer import Trainer

try:
    __version__ = version("rlstack")
except PackageNotFoundError:
    pass
