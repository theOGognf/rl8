"""Skip connection module definitions."""

import torch
import torch.nn as nn

from ..functional import skip_connection
from .module import Module


class SequentialSkipConnection(Module[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """Sequential skip connection.

    Apply a skip connection to an input and the output of a layer that
    uses that input.

    Args:
        embed_dim: Original input feature size.
        kind: Type of skip connection to apply.
            Options include:

                - "residual" for a standard residual connection (summing outputs)
                - "cat" for concatenating outputs
                - `None` for no skip connection

        fan_in: Whether to apply a linear layer after each skip connection
            automatically such that the output of the forward pass will
            always have dimension ``embed_dim``.

    """

    #: Number of input features for each module.
    _in_features: list[int]

    #: Modules associated with the sequential forward passes.
    _layers: nn.ModuleList

    #: Kind of skip connection. "residual" for a standard residual connection
    #: (summing outputs), "cat" for concatenating outputs, and ``None`` for no
    #: skip connection (reduces to a regular, sequential module).
    kind: None | str

    def __init__(self, embed_dim: int, kind: None | str = "cat") -> None:
        super().__init__()
        self._in_features = [embed_dim]
        self._layers = torch.nn.ModuleList([])
        self.kind = kind
        match kind:
            case "cat":
                self._layers.append(nn.Linear(self._skip_features, embed_dim))
            case _:
                self._layers.append(nn.Identity())

    @property
    def _skip_features(self) -> int:
        """Return the number of output features according to the number of input
        features and the kind of skip connection.

        """
        match self.kind:
            case "residual":
                return self._in_features[-1]
            case "cat":
                return 2 * self._in_features[-1]
            case None:
                return self._in_features[-1]
        raise ValueError(f"No skip connection type for {self.kind}.")

    def append(self, module: nn.Module, /) -> int:
        """Append ``module`` to the skip connection.

        If :attr:`SequentialSkipConnection.fan_in` is ``True``, then a
        fan-in layer is also appended after ``module`` to reduce the
        number of output features back to the input dimension.

        Args:
            module: Module to append and apply a skip connection to.

        Returns:
            Number of output features from the sequential skip connection.

        """
        self._in_features.append(self._skip_features)
        self._layers.append(module)
        match self.kind:
            case "cat":
                linear = nn.Linear(self._in_features[-1], self._in_features[0])
                self._in_features.append(linear.out_features)
                self._layers.append(linear)
            case _:
                self._in_features.append(self._in_features[-1])
                self._layers.append(nn.Identity())
        return self.out_features

    def forward(self, x: torch.Tensor, y: torch.Tensor, /) -> torch.Tensor:
        """Perform a sequential skip connection, first applying a skip
        connection to ``x`` and ``y``, and then sequentially applying skip
        connections to the output and the output of the next layer.

        Args:
            x: Skip connection seed with shape ``[B, T, ...]``.
            y: Skip connection seed with same shape as ``y``.

        Returns:
            A tensor with shape depending on
            :attr:`SequentialSkipConnection.fan_in` and
            :attr:`SequentialSkipConnection.kind`.

        """
        y = skip_connection(x, y, kind=self.kind)
        for i, layer in enumerate(self._layers):
            if i % 2:
                y = skip_connection(y, layer(y), kind=self.kind)
            else:
                y = layer(y)
        return y

    @property
    def in_features(self) -> int:
        """Return the first number of input features."""
        return self._in_features[0]

    @property
    def out_features(self) -> int:
        """Return the number of output features according to the number of input
        features, the kind of skip connection, and whether there's a fan-in
        layer.

        """
        match self.kind:
            case "cat":
                return self._in_features[0]
            case _:
                return self._skip_features
