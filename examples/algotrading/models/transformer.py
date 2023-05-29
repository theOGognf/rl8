from typing import Any

from rlstack import Model
from rlstack.data import DataKeys
from rlstack.specs import TensorSpec
from rlstack.views import ViewRequirement


class Transformer(Model):
    def __init__(
        self, observation_spec: TensorSpec, action_spec: TensorSpec, /, **config: Any
    ) -> None:
        super().__init__(observation_spec, action_spec, **config)
        config.get("embed_dim", 8)
        seq_len = config.get("seq_len", 16)
        self.view_requirements = {
            (DataKeys.OBS, "LOG_CHANGE(price)"): ViewRequirement(shift=seq_len)
        }
