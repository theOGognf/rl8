import math
from enum import IntEnum
from typing import Any

import torch
from tensordict import TensorDict

from rlstack.data import Device
from rlstack.env import Env
from rlstack.specs import UnboundedContinuousTensorSpec


class Action(IntEnum):
    """Enumeration for environment actions just for readability."""

    HOLD = 0

    BUY = 1

    SELL = 2


class MockTrader(Env):
    """An environment that mocks algotrading.

    An asset's price is simulated according to the following equation
    ``y = k * t * sin(f * t)`` where ``k``, ``f`` and ``t0`` are all
    randomly sampled from their own independent uniform distributions
    defined by bounds specified in ``config``.

    A policy must learn to hold, buy, or sell the asset based on the
    asset's change in price with respect to the previous day and with
    respect to the price at which the policy had previously bought the
    asset.

    This environment serves as a playground for different kinds of models.
    Feedforward models could specify view requirements to utilize aggregated
    metrics or sequence-based models, while recurrent models could accept
    the environment's observations as-is to learn complex trading behaviors.
    The environment also returns action masks that can be used for defining
    custom action distributions that could help accelerate learning.



    """

    def __init__(
        self,
        num_envs: int,
        /,
        *,
        config: dict[str, Any] | None = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(num_envs, config=config, device=device)
        self.max_horizon = 128
        self.observation_spec = UnboundedContinuousTensorSpec(1, device=device)
        self.slope_bounds = self.config.get("slope_bounds", 1)
        self.frequency_bounds = self.config.get("frequency_bounds", math.pi)
        self.reset(config=self.config)

    def reset(self, *, config: dict[str, Any] | None = None) -> TensorDict:
        config = config or {}
        self.slope_bounds = config.get("slope_bounds", self.slope_bounds)
        self.frequency_bounds = config.get("frequency_bounds", self.frequency_bounds)
        t = torch.randint(0, 10, size=(self.num_envs, 1), device=self.device)
        slopes = (
            -self.slope_bounds * torch.rand(self.num_envs, 1, device=self.device)
            + self.slope_bounds
        )
        frequencies = self.frequency_bounds * torch.rand(
            self.num_envs, 1, device=self.device
        )
        self.state = TensorDict(
            {
                "invested": torch.zeros(self.num_envs, 1, device=self.device).bool(),
                "invested_price": torch.zeros(
                    self.num_envs, 1, device=self.device
                ).float(),
                "frequencies": frequencies,
                "slopes": slopes,
                "t": t,
                "price": t * slopes * torch.sin(t * frequencies),
                "LOG_CHANGE(price)": torch.zeros(
                    self.num_envs, 1, device=self.device
                ).float(),
                "LOG_CHANGE(price, invested_price)": torch.zeros(
                    self.num_envs, 1, device=self.device
                ).float(),
            },
            batch_size=self.num_envs,
            device=self.device,
        )
        return self.state.select(
            "invested", "LOG_CHANGE(price)", "LOG_CHANGE(price, invested_price)"
        )

    def step(self, action: torch.Tensor) -> TensorDict:
        old_price = self.state["price"].clone()

        # Handle buy actions
        buy_mask = action == Action.BUY
        self.state["invested"][buy_mask] = 1
        self.state["invested_price"][buy_mask] = old_price[buy_mask]

        # Handle sell actions
        sell_mask = action == Action.SELL
        self.state["invested"][sell_mask] = 0

        # Handle hold actions
        not_invested_mask = self.state["invested"] == 0
        self.state["invested_price"][not_invested_mask] = old_price[not_invested_mask]

        # Main environment state update
        self.state["t"] += 1
        self.state["price"] = (
            self.state["t"]
            * self.state["slopes"]
            * torch.sin(self.state["t"] * self.state["frequencies"])
        )
        self.state["LOG_CHANGE(price)"] = torch.log(self.state["price"]) - torch.log(
            old_price
        )
        self.state["LOG_CHANGE(price, invested_price)"] = torch.log(
            self.state["price"]
        ) - torch.log(self.state["invested_price"])
        obs = self.state.select(
            "invested",
            "LOG_CHANGE(price)",
            "LOG_CHANGE(price, invested_price)",
        )
        return
