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


class AttentiveAlpaca(Model):
    """A model that applies self-attention to historical price changes
    to eventually construct logits used for sampling actions.

    The model eventually reduces the environment observation into a 1D
    tensor that's fed into feature and value function models that produce
    the model's outputs. The feature model and value function
    model share parameters since they share the same input from the
    feature vector created partly from the self-attention mechanism.

    Args:
        observation_spec: Environment observation spec.
        action_spec: Environment action spec.
        invested_embed_dim: The size of the embedding to create for the
            environment observation indicating whether the policy is
            already invested in the asset.
        price_embed_dim: The size of the embedding for historical price
            changes.
        seq_len: Number of historical price changes to use for the
            self-attention mechanism. This should always be less than
            the environment horizon used during training.
        num_heads: Number of attention heads to use per self-attention
            layer.
        num_layers: Number of self-attention layers to use.
        hiddens: Hidden neurons for each layer in the feature and value
            function models. The first element is also used as the number
            of hidden neurons in the self-attention mechanism.
        activation_fn: Activation function used by all components.
        bias: Whether to use a bias in the linear layers for the feature
            and value function models.

    """

    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        invested_embed_dim: int = 2,
        price_embed_dim: int = 8,
        seq_len: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        hiddens: tuple[int, ...] = (64, 64),
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
            share_parameters=True,
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
        feature_head = nn.Linear(hiddens[-1], 3)
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
            ],
        )
        x_price = masked_avg(
            x_price,
            mask=~batch[DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.PADDING_MASK],
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
