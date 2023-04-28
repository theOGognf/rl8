"""Top-level package interface."""

from importlib.metadata import PackageNotFoundError, version

from .algorithm import Algorithm
from .env import Env, GenericEnv
from .policy import Distribution, GenericModel, Model, Policy
from .trainer import Trainer

try:
    __version__ = version("rlstack")
except PackageNotFoundError:
    pass
