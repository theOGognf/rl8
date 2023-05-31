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
    ``y[k + 1] = (1 + km) * (1 + kc * sin(f * t)) * y[k]`` where
    ``km``, ``kc``, ``f``, and ``y[0]`` are all randomly sampled
    from their own independent uniform distributions, some of which
    are defined by values in ``config``.

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
                    2, shape=torch.Size([3]), device=device, dtype=torch.bool
                ),
                "invested": DiscreteTensorSpec(
                    2, shape=torch.Size([1]), device=device, dtype=torch.long
                ),
                "LOG_CHANGE(price)": UnboundedContinuousTensorSpec(
                    1, device=device, dtype=torch.float32
                ),
                "LOG_CHANGE(price, invested_price)": UnboundedContinuousTensorSpec(
                    1, device=device, dtype=torch.float32
                ),
            }
        )  # type: ignore[no-untyped-call]
        self.action_spec = DiscreteTensorSpec(3, shape=torch.Size([1]), device=device)
        self.f_bounds = self.config.get("f_bounds", math.pi)
        self.k_cyclic_bounds = self.config.get("k_cyclic_bounds", 0.05)
        self.k_market_bounds = self.config.get("k_market_bounds", 0.05)

    def reset(self, *, config: dict[str, Any] | None = None) -> TensorDict:
        config = config or {}
        self.f_bounds = self.config.get("f_bounds", self.f_bounds)
        self.k_cyclic_bounds = config.get("k_cyclic_bounds", self.k_cyclic_bounds)
        self.k_market_bounds = config.get("k_market_bounds", self.k_market_bounds)
        f = torch.empty(self.num_envs, 1, device=self.device).uniform_(0, self.f_bounds)
        k_cyclic = torch.empty(self.num_envs, 1, device=self.device).uniform_(
            -self.k_cyclic_bounds, self.k_cyclic_bounds
        )
        k_market = torch.empty(self.num_envs, 1, device=self.device).uniform_(
            -self.k_market_bounds, self.k_market_bounds
        )
        t = torch.randint(0, 10, size=(self.num_envs, 1), device=self.device)
        price = torch.empty(self.num_envs, 1, device=self.device).uniform_(100, 10000)
        action_mask = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.bool
        )
        action_mask[:, Action.HOLD] = True
        action_mask[:, Action.BUY] = True
        action_mask[:, Action.SELL] = False
        self.state = TensorDict(
            {
                "action_mask": action_mask,
                "invested": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.long
                ),
                "invested_price": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.float32
                ),
                "f": f,
                "k_cyclic": k_cyclic,
                "k_market": k_market,
                "t": t,
                "price": price,
                "LOG_CHANGE(price)": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.float32
                ),
                "LOG_CHANGE(price, invested_price)": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.float32
                ),
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
        reward = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

        # Handle buy actions
        buy_mask = (action == Action.BUY).flatten()
        self.state["invested"][buy_mask] = 1
        self.state["invested_price"][buy_mask] = old_price[buy_mask]

        # Handle sell actions
        sell_mask = (action == Action.SELL).flatten()
        self.state["invested"][sell_mask] = 0
        reward[sell_mask] = torch.log(old_price[sell_mask]) - torch.log(
            self.state["invested_price"][sell_mask]
        )

        # Handle hold actions
        hold_mask = (action == Action.HOLD).flatten()
        invested_mask = (self.state["invested"] == 1).flatten()
        not_invested_mask = ~invested_mask
        self.state["invested_price"][not_invested_mask] = old_price[not_invested_mask]
        reward[invested_mask & hold_mask] = self.state["LOG_CHANGE(price)"][
            invested_mask & hold_mask
        ].clone()

        # Main environment state update
        self.state["action_mask"][invested_mask, Action.HOLD] = True
        self.state["action_mask"][invested_mask, Action.BUY] = False
        self.state["action_mask"][invested_mask, Action.SELL] = True
        self.state["action_mask"][not_invested_mask, Action.HOLD] = True
        self.state["action_mask"][not_invested_mask, Action.BUY] = True
        self.state["action_mask"][not_invested_mask, Action.SELL] = False
        self.state["t"] += 1
        self.state["price"] *= (1 + self.state["k_market"]) * (
            1 + self.state["k_cyclic"] * torch.sin(self.state["t"] * self.state["f"])
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
