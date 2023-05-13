"""Top-level package interface."""

from importlib.metadata import PackageNotFoundError, version

from .algorithms import Algorithm
from .distributions import Categorical, Distribution, Normal, SquashedNormal
from .env import Env, GenericEnv
from .models import DefaultContinuousModel, DefaultDiscreteModel, Model
from .policies import Policy
from .trainer import Trainer

try:
    __version__ = version("rlstack")
except PackageNotFoundError:
    pass
