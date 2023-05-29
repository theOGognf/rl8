import math
from enum import IntEnum
from typing import Any

import torch
from tensordict import TensorDict

from rlstack import Env
from rlstack.data import DataKeys, Device
from rlstack.specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)


class Action(IntEnum):
    """Enumeration of environment actions for readability."""

    HOLD = 0

    BUY = 1

    SELL = 2


class AlgoTrading(Env):
    """An environment that mocks algotrading.

    An asset's price is simulated according to the equation
    ``y = k * t * sin(f * t)`` where ``k``, ``f``, and ``t0`` are all
    randomly sampled from their own independent uniform distributions
    defined by bounds specified in ``config``.

    A policy must learn to hold, buy, or sell the asset based on the
    asset's change in price with respect to the previous day and with
    respect to the price at which the policy had previously bought the
    asset.

    This environment serves as a playground for different kinds of models.
    Feedforward models could specify view requirements to utilize aggregated
    metrics or sequence-based components, while recurrent models could accept
    the environment's observations as-is.

    """

    # Environment state that's reset when the environment is reset and is
    # updated when the environment is stepped.
    state: TensorDict

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
        self.observation_spec = CompositeSpec(
            {
                "action_mask": DiscreteTensorSpec(
                    2, shape=torch.Size([3]), device=device
                ),
                "invested": DiscreteTensorSpec(2, shape=torch.Size([1]), device=device),
                "LOG_CHANGE(price)": UnboundedContinuousTensorSpec(1, device=device),
                "LOG_CHANGE(price, invested_price)": UnboundedContinuousTensorSpec(
                    1, device=device
                ),
            }
        )  # type: ignore[no-untyped-call]
        self.action_spec = DiscreteTensorSpec(3, shape=torch.Size([1]), device=device)
        self.slope_bounds = self.config.get("slope_bounds", 1)
        self.frequency_bounds = self.config.get("frequency_bounds", math.pi)

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
        action_mask = torch.zeros(self.num_envs, 3, device=self.device).bool()
        action_mask[:, Action.HOLD] = 1
        action_mask[:, Action.BUY] = 1
        action_mask[:, Action.SELL] = 0
        self.state = TensorDict(
            {
                "action_mask": action_mask,
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
            "action_mask",
            "invested",
            "LOG_CHANGE(price)",
            "LOG_CHANGE(price, invested_price)",
        )

    def step(self, action: torch.Tensor) -> TensorDict:
        old_price = self.state["price"].clone()
        reward = torch.zeros(self.num_envs, 1, device=self.device).float()

        # Handle buy actions
        buy_mask = action == Action.BUY
        self.state["invested"][buy_mask] = 1
        self.state["invested_price"][buy_mask] = old_price[buy_mask]

        # Handle sell actions
        sell_mask = action == Action.SELL
        self.state["invested"][sell_mask] = 0
        reward[sell_mask] = torch.log(old_price[sell_mask]) - torch.log(
            self.state["invested_price"][sell_mask]
        )

        # Handle hold actions
        invested_mask = self.state["invested"] == 1
        not_invested_mask = ~invested_mask
        self.state["invested_price"][not_invested_mask] = old_price[not_invested_mask]

        # Main environment state update
        self.state["action_mask"][invested_mask][:, Action.HOLD] = 1
        self.state["action_mask"][invested_mask][:, Action.BUY] = 0
        self.state["action_mask"][invested_mask][:, Action.SELL] = 1
        self.state["action_mask"][not_invested_mask][:, Action.HOLD] = 1
        self.state["action_mask"][not_invested_mask][:, Action.BUY] = 1
        self.state["action_mask"][not_invested_mask][:, Action.SELL] = 0
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
            "action_mask",
            "invested",
            "LOG_CHANGE(price)",
            "LOG_CHANGE(price, invested_price)",
        )
        return TensorDict(
            {DataKeys.OBS: obs, DataKeys.REWARDS: reward},
            batch_size=self.num_envs,
            device=self.device,
        )
