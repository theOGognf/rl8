import torch

from rlstack import Algorithm, Trainer
from rlstack.conditions import Plateaus

from .env import AlgoTrading
from .models import MischievousMule

trainer = Trainer(
    AlgoTrading,
    algorithm_cls=Algorithm,
    algorithm_config={
        "model_cls": MischievousMule,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "enable_amp": True,
    },
    stop_conditions=[Plateaus("returns/mean", patience=10, rtol=0.05)],
)
trainer.run()
