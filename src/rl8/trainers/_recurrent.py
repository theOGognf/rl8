"""High-level training interfaces."""

from ..algorithms import RecurrentAlgorithm
from ._base import GenericTrainerBase


class RecurrentTrainer(GenericTrainerBase[RecurrentAlgorithm]):
    """Higher-level training interface that interops with other tools for
    tracking and saving experiments (i.e., MLflow).

    This is the preferred training interface when training recurrent
    policies in most cases.

    """
