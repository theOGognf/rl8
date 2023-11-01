import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import TensorSpec

from rl8 import Model
from rl8.data import DataKeys
from rl8.nn import MLP, get_activation
from rl8.views import ViewRequirement

FINFO = torch.finfo()


class MischievousMule(Model):
    """A model that aggregates historical price changes at different
    intervals to form a latent vector that's fed into other model
    components.

    The feature model and value function model share parameters since
    they share the same input from the feature vector created partly
    from the aggregation mechanism.

    Args:
        observation_spec: Environment observation spec.
        action_spec: Environment action spec.
        invested_embed_dim: The size of the embedding to create for the
            environment observation indicating whether the policy is
            already invested in the asset.
        seq_len: Number of historical price changes to use for the
            aggregation mechanism. This should always be less than
            the environment horizon used during training.
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
        seq_len: int = 4,
        hiddens: tuple[int, ...] = (128, 128),
        activation_fn: str = "relu",
    ) -> None:
        super().__init__(
            observation_spec,
            action_spec,
            invested_embed_dim=invested_embed_dim,
            seq_len=seq_len,
            hiddens=hiddens,
            activation_fn=activation_fn,
        )
        assert not seq_len % 4, "`seq_len` must be a factor of 4 for this model."
        self.seq_len = seq_len
        # Feedforward models use a default view requirement that passes
        # only the most recent observation to the model for inference.
        # We specify a view requirement on historical price changes by
        # adding a nested key to the default view requirement. This
        # keeps the default view requirement while also allowing the model
        # to use historical price changes as additional inputs.
        self.view_requirements[(DataKeys.OBS, "LOG_CHANGE(price)")] = ViewRequirement(
            shift=seq_len
        )
        self.invested_embedding = nn.Embedding(2, invested_embed_dim)
        self.feature_model = nn.Sequential(
            MLP(
                invested_embed_dim + 5,
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
                invested_embed_dim + 5,
                hiddens,
                activation_fn=activation_fn,
                norm_layer=nn.BatchNorm1d,
                inplace=False,
            ),
            get_activation(activation_fn),
            nn.Linear(hiddens[-1], 1),
        )
        self._value = None

    def forward(self, batch: TensorDict, /) -> TensorDict:
        x_invested = self.invested_embedding(batch[DataKeys.OBS, "invested"].flatten())
        x_price = batch[DataKeys.OBS, "LOG_CHANGE(price)", DataKeys.INPUTS]
        x_price = torch.cat(
            [
                torch.sum(x_price[:, : (self.seq_len // 4), ...], dim=1),
                torch.sum(x_price[:, : (self.seq_len // 2), ...], dim=1),
                torch.sum(x_price[:, -(self.seq_len // 2) :, ...], dim=1),
                torch.sum(x_price[:, -(self.seq_len // 4) :, ...], dim=1),
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                x_invested,
                batch[DataKeys.OBS, "LOG_CHANGE(price, position)"],
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
