"""High-level training interfaces."""

from ..algorithms import Algorithm
from ._base import GenericTrainerBase


class Trainer(GenericTrainerBase[Algorithm]):
    """Higher-level training interface that interops with other tools for
    tracking and saving experiments (i.e., MLflow).

    This is the preferred training interface when training feedforward
    (i.e., non-recurrent) policies in most cases.

    """
