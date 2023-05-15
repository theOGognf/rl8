"""Definitions related to PPO algorithms (data collection and training steps).

Algorithms assume environments are parallelized much like
`IsaacGym environments`_ and are infinite horizon with no terminal
conditions. These assumptions allow the learning procedure to occur
extremely fast even for complex, sequence-based models because:

    - Environments occur in parallel and are batched into a contingous
      buffer.
    - All environments are reset in parallel after a predetermined
      horizon is reached.
    - All operations occur on the same device, removing overhead
      associated with data transfers between devices.

.. _`IsaacGym environments`: https://arxiv.org/pdf/2108.10470.pdf

"""

from ._feedforward import Algorithm
from ._recurrent import RecurrentAlgorithm

__all__ = ["Algorithm", "RecurrentAlgorithm"]
