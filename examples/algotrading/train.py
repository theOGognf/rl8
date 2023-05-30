from rlstack import Trainer

from .env import AlgoTrading
from .models import Transformer

trainer = Trainer(
    AlgoTrading, algorithm_config={"model_cls": Transformer, "entropy_coeff": 1e-3}
)
trainer.run()
