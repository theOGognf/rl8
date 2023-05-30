import torch
import torch.nn as nn
from tensordict import TensorDict

from rlstack import Model
from rlstack.data import DataKeys
from rlstack.nn import (
    MLP,
    SelfAttention,
    SelfAttentionStack,
    get_activation,
    masked_avg,
)
from rlstack.specs import TensorSpec
from rlstack.views import ViewRequirement

FINFO = torch.finfo()


class Transformer(Model):
    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        invested_embed_dim: int = 4,
        price_embed_dim: int = 8,
        seq_len: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        hiddens: tuple[int, ...] = (256, 256),
        activation_fn: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__(
            observation_spec,
            action_spec,
            invested_embed_dim=invested_embed_dim,
            price_embed_dim=price_embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
            hiddens=hiddens,
            activation_fn=activation_fn,
            bias=bias,
        )
        self.view_requirements[(DataKeys.OBS, "LOG_CHANGE(price)")] = ViewRequirement(
            shift=seq_len
        )
        self.invested_embedding = nn.Embedding(2, invested_embed_dim)
        self.price_embedding = nn.Linear(1, price_embed_dim)
        self.price_attention = SelfAttentionStack(
            SelfAttention(
                price_embed_dim,
                num_heads=num_heads,
                hidden_dim=hiddens[0],
                activation_fn=activation_fn,
                skip_kind="residual",
            ),
            num_layers,
        )
        self.feature_model = nn.Sequential(
            MLP(
                invested_embed_dim + 1 + price_embed_dim,
                hiddens,
                activation_fn=activation_fn,
                bias=bias,
            ),
            get_activation(activation_fn),
        )
        feature_head = nn.Linear(
            hiddens[-1], action_spec.shape[0] * action_spec.space.n
        )
        nn.init.uniform_(feature_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(feature_head.bias)
        self.feature_model.append(feature_head)
        self.vf_model = nn.Sequential(
            MLP(
                invested_embed_dim + 1 + price_embed_dim,
                hiddens,
                activation_fn=activation_fn,
                bias=bias,
            ),
            get_activation(activation_fn),
            nn.Linear(hiddens[-1], 1),
        )
        self._value = None

    def forward(self, batch: TensorDict, /) -> TensorDict:
        x_invested = self.invested_embedding(batch[DataKeys.OBS, "invested"].flatten())
        x_price = self.price_embedding(
            batch[DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.INPUTS]
        )
        x_price = self.price_attention(
            x_price,
            key_padding_mask=batch[
                DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.PADDING_MASK
            ].bool(),
        )
        x_price = masked_avg(
            x_price,
            mask=~batch[
                DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.PADDING_MASK
            ].bool(),
            dim=1,
            keepdim=False,
        )
        x = torch.cat(
            [
                x_invested,
                batch[DataKeys.OBS, "LOG_CHANGE(price, invested_price)"],
                x_price,
            ],
            dim=-1,
        )
        features = self.feature_model(x).reshape(-1, 1, 3)
        inf_mask = torch.clamp(
            torch.log(batch[DataKeys.OBS, "action_mask"]), min=FINFO.min, max=FINFO.max
        ).reshape(-1, 1, 3)
        masked_logits = features + inf_mask
        self._value = self.vf_model(x)
        return TensorDict(
            {"logits": masked_logits},
            batch_size=batch.batch_size,
            device=batch.device,
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value
