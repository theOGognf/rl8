import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec

from rl8 import RecurrentModel
from rl8.data import DataKeys
from rl8.nn import MLP, get_activation

FINFO = torch.finfo()


class LazyLemur(RecurrentModel):
    """An LSTM model that maintains states across horizons.

    Args:
        observation_spec: Environment observation spec.
        action_spec: Environment action spec.
        invested_embed_dim: The size of the embedding to create for the
            environment observation indicating whether the policy is
            already invested in the asset.
        hidden_size: Hidden neurons within the LSTM.
        num_layers: Number of LSTM cells.
        hiddens: Hidden neurons for each layer in the feature and value
            function models.
        activation_fn: Activation function used by all components.

    """

    def __init__(
        self,
        observation_spec: TensorSpec,
        action_spec: TensorSpec,
        /,
        invested_embed_dim: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        hiddens: tuple[int, ...] = (64, 64),
        activation_fn: str = "relu",
    ) -> None:
        super().__init__(
            observation_spec,
            action_spec,
            invested_embed_dim=invested_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.state_spec = CompositeSpec(
            {
                DataKeys.HIDDEN_STATES: UnboundedContinuousTensorSpec(
                    shape=torch.Size([num_layers, hidden_size]),
                    device=action_spec.device,
                ),
                DataKeys.CELL_STATES: UnboundedContinuousTensorSpec(
                    shape=torch.Size([num_layers, hidden_size]),
                    device=action_spec.device,
                ),
            }
        )
        self.invested_embedding = nn.Embedding(2, invested_embed_dim)
        self.lstm = nn.LSTM(
            invested_embed_dim + 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.feature_model = nn.Sequential(
            MLP(
                hidden_size,
                hiddens,
                activation_fn=activation_fn,
                norm_layer=nn.BatchNorm1d,
                inplace=False,
            ),
            get_activation(activation_fn),
        )
        feature_head = nn.Linear(hiddens[-1], 3)
        nn.init.uniform_(feature_head.weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(feature_head.bias)
        self.feature_model.append(feature_head)
        self.vf_model = nn.Sequential(
            MLP(
                hidden_size,
                hiddens,
                activation_fn=activation_fn,
                norm_layer=nn.BatchNorm1d,
                inplace=False,
            ),
            get_activation(activation_fn),
            nn.Linear(hiddens[-1], 1),
        )
        self._value = None

    def forward(
        self, batch: TensorDict, states: TensorDict, /
    ) -> tuple[TensorDict, TensorDict]:
        B, T = batch.shape[:2]
        x_invested = self.invested_embedding(
            batch[DataKeys.OBS, "invested"].flatten()
        ).reshape(B, T, -1)
        x = torch.cat(
            [
                x_invested,
                batch[DataKeys.OBS, "LOG_CHANGE(price, position)"],
                batch[DataKeys.OBS, "LOG_CHANGE(price)"],
            ],
            dim=-1,
        )
        h_0 = states[DataKeys.HIDDEN_STATES][:, 0, ...].permute(1, 0, 2).contiguous()
        c_0 = states[DataKeys.CELL_STATES][:, 0, ...].permute(1, 0, 2).contiguous()
        latents, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        latents = latents.reshape(B * T, -1)
        features = self.feature_model(latents).reshape(-1, 1, 3)
        self._value = self.vf_model(latents).reshape(-1, 1)
        inf_mask = torch.clamp(
            torch.log(batch[DataKeys.OBS, "action_mask"]), min=FINFO.min, max=FINFO.max
        ).reshape(-1, 1, 3)
        masked_logits = features + inf_mask
        self._value = self.vf_model(latents).reshape(-1, 1)
        return TensorDict(
            {"logits": masked_logits},
            batch_size=masked_logits.size(0),
            device=batch.device,
        ), TensorDict(
            {
                DataKeys.HIDDEN_STATES: h_n.permute(1, 0, 2),
                DataKeys.CELL_STATES: c_n.permute(1, 0, 2),
            },
            batch_size=batch.size(0),
        )

    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value
