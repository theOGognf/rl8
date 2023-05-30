from rlstack import Trainer
from rlstack.conditions import Plateaus

from .env import AlgoTrading
from .models import Transformer

trainer = Trainer(
    AlgoTrading,
    algorithm_config={"model_cls": Transformer},
    stop_conditions=[Plateaus("returns/mean", rtol=0.05)],
)
trainer.run()
