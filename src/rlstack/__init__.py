"""Top-level package interface."""

from importlib.metadata import PackageNotFoundError, version

from .algorithms import Algorithm, GenericAlgorithmBase, RecurrentAlgorithm
from .distributions import Categorical, Distribution, Normal, SquashedNormal
from .env import Env, GenericEnv
from .models import (
    DefaultContinuousModel,
    DefaultContinuousRecurrentModel,
    DefaultDiscreteModel,
    DefaultDiscreteRecurrentModel,
    GenericModel,
    GenericModelBase,
    GenericRecurrentModel,
    Model,
    RecurrentModel,
)
from .policies import (
    GenericPolicyBase,
    MLflowPolicyModel,
    MLflowRecurrentPolicyModel,
    Policy,
    RecurrentPolicy,
)
from .trainers import GenericTrainerBase, RecurrentTrainer, Trainer

try:
    __version__ = version("rlstack")
except PackageNotFoundError:
    pass
