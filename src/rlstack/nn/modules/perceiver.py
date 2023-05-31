"""Perceiver definitions."""

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention, SelfAttentionStack
from .module import Module


class PerceiverLayer(
    Module[
        [torch.Tensor, torch.Tensor, None | torch.Tensor, None | torch.Tensor],
        torch.Tensor,
    ]
):
    """An implementation of a `Perceiver`_ with cross-attention followed
    by self-attention stacks.

    Useful for embedding several, variable-length sequences into a latent
    array for dimensionality reduction. Allows inputs of different feature
    sizes to be embedded into a constant size.

    Args:
        embed_dim: Feature dimension of the latent array and input sequence.
            Each sequence is expected to be embedded by its own embedder, which
            could just be a simple linear transform.
        num_heads: Number of attention heads in the cross-attention and
            self-attention modules.
        hidden_dim: Number of hidden features in the hidden layers
            of the feedforward networks that're after performing attention.
        activation_fn: Activation function ID.
        attention_dropout: Sequence dropout in the attention heads.
        hidden_dropout: Feedforward dropout after performing attention.
        skip_kind: Kind of residual or skip connection to make between
            outputs of the multihead attentions and the feedforward
            modules.
        share_parameters: Whether to use the same parameters for the layers
            in the self-attention stack.

    .. _`Perceiver`: https://arxiv.org/pdf/2103.03206.pdf

    """

    def __init__(
        self,
        embed_dim: int,
        /,
        *,
        num_heads: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation_fn: str = "relu",
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        skip_kind: str = "cat",
        share_parameters: bool = False,
    ) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(
            embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            skip_kind=skip_kind,
        )
        self.self_attention = SelfAttentionStack(
            SelfAttention(
                embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                activation_fn=activation_fn,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                skip_kind=skip_kind,
            ),
            num_layers,
            share_parameters=share_parameters,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply cross-attention keys to a query, mapping the keys of
        sequence length ``K`` to the query of sequence length ``Q``.

        Args:
            q: Query with shape ``[B, Q, E]``. Usually the latent array from
                previous forward passes or perceiver layers.
            kv: Keys with shape ``[B, K, E]``.
            key_padding_mask: Mask with shape ``[B, K]`` indicating sequence
                elements of ``kv`` that are PADDED or INVALID values.
            attention_mask: Mask with shape ``[Q, K]`` that indicates whether
                elements in ``Q`` can attend to elements in ``K``.

        Returns:
            Values with shape ``[B, Q, E]``.

        """
        latent = self.cross_attention(q, kv, key_padding_mask, attention_mask)
        return self.self_attention(latent, None, None)


class PerceiverIOLayer(
    Module[
        [torch.Tensor, torch.Tensor, None | torch.Tensor, None | torch.Tensor],
        torch.Tensor,
    ]
):
    """An implementation of `PerceiverIO`_ with cross-attention followed by
    self-attention stacks followed by cross-attention with a fixed-sized
    output array.

    In addition to the benefits of `PerceiverLayer`, this module attends a
    latent array to a final output dimensionality to effectively apply
    weighted averaging of sequences to a different dimension. Useful if the
    latent array needs to be processed into several, different-sized
    sequences for separate outputs.

    Args:
        embed_dim: Feature dimension of the latent array and input sequence.
            Each sequence is expected to be embedded by its own embedder, which
            could just be a simple linear transform.
        output_seq_dim: Output sequence size to transform the latent array
            sequence size to.
        num_heads: Number of attention heads in the cross-attention and
            self-attention modules.
        hidden_dim: Number of hidden features in the hidden layers
            of the feedforward networks that're after performing attention.
        activation_fn: Activation function ID.
        attention_dropout: Sequence dropout in the attention heads.
        hidden_dropout: Feedforward dropout after performing attention.
        skip_kind: Kind of residual or skip connection to make between
            outputs of the multihead attentions and the feedforward
            modules.
        share_parameters: Whether to use the same parameters for the layers
            in the self-attention stack.

    .. _`PerceiverIO`: https://arxiv.org/pdf/2107.14795.pdf

    """

    def __init__(
        self,
        embed_dim: int,
        output_seq_dim: int,
        /,
        *,
        num_heads: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation_fn: str = "relu",
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        skip_kind: str = "cat",
        share_parameters: bool = False,
    ) -> None:
        super().__init__()
        self.perceiver_layer = PerceiverLayer(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            activation_fn=activation_fn,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            skip_kind=skip_kind,
            share_parameters=share_parameters,
        )
        self.output_query = nn.Parameter(torch.zeros([output_seq_dim, embed_dim]))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.output_query)
        self.decoder = CrossAttention(
            embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            skip_kind=skip_kind,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: None | torch.Tensor = None,
        attention_mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply cross-attention keys to a query, mapping the keys of
        sequence length ``K`` to the query of sequence length ``Q``.

        Args:
            q: Query with shape ``[B, Q, E]``. Usually the latent array from
                previous forward passes or perceiver layers.
            kv: Keys with shape ``[B, K, E]``.
            key_padding_mask: Mask with shape ``[B, K]`` indicating sequence
                elements of ``kv`` that are PADDED or INVALID values.
            attention_mask: Mask with shape ``[Q, K]`` that indicates whether
                elements in ``Q`` can attend to elements in ``K``.

        Returns:
            Values with shape ``[B, O, E]`` where ``O`` is the output array sequence
            size.

        """
        B = q.size(0)
        output_query = self.output_query.unsqueeze(0).expand(
            B, *self.output_query.shape
        )
        latent = self.perceiver_layer(q, kv, key_padding_mask, attention_mask)
        return self.decoder(output_query, latent, None, None)
