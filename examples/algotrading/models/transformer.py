from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rlstack import Model
from rlstack.data import DataKeys
from rlstack.nn import SelfAttention, SelfAttentionStack, masked_avg
from rlstack.specs import TensorSpec
from rlstack.views import ViewRequirement


class Transformer(Model):
    def __init__(
        self, observation_spec: TensorSpec, action_spec: TensorSpec, /, **config: Any
    ) -> None:
        super().__init__(observation_spec, action_spec, **config)
        embed_dim = config.get("embed_dim", 8)
        seq_len = config.get("seq_len", 4)
        num_layers = config.get("num_layers", 2)
        self.view_requirements[(DataKeys.OBS, "LOG_CHANGE(price)")] = ViewRequirement(
            shift=seq_len
        )
        self.invested_embedding = nn.Embedding(2, embed_dim)
        self.price_embedding = nn.Linear(1, embed_dim)
        self.price_attention = SelfAttentionStack(SelfAttention(embed_dim), num_layers)
        self.feature_net = nn.Sequential()

    def forward(self, batch: TensorDict, /) -> TensorDict:
        x_invested = self.invested_embedding(batch[DataKeys.OBS, "invested"])
        x_price = self.price_embedding(
            batch[DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.INPUTS]
        )
        x_price = self.price_attention(
            x_price,
            key_padding_mask=batch[
                DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.PADDING_MASK
            ],
        )
        x_price = masked_avg(
            x_price,
            mask=~batch[DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.PADDING_MASK],
            dim=1,
            keepdim=False,
        )
        torch.cat(
            [
                x_invested,
                batch[DataKeys.OBS, "LOG_CHANGE(price, invested_price)"],
                x_price,
            ]
        )
