from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec

from rl8 import Env
from rl8.data import DataKeys, Device


class Pendulum(Env):
    """Reimplementation of the classic `Pendulum`_ environment.

    .. _`Pendulum`: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py

    """

    # Max number of steps that can be taken with the environment before
    # requiring a reset.
    max_horizon = 512

    # Environment state that's reset when the environment is reset and is
    # updated when the environment is stepped.
    state: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        /,
        horizon: None | int = None,
        *,
        config: dict[str, Any] | None = None,
        device: Device = "cpu",
    ):
        super().__init__(num_envs, horizon, config=config, device=device)
        self.max_speed = self.config.get("max_speed", 8)
        self.max_torque = self.config.get("max_torque", 2.0)
        self.dt = self.config.get("dt", 0.05)
        self.g = self.config.get("g", 10.0)
        self.m = self.config.get("m", 1.0)
        self.l = self.config.get("l", 1.0)

        self.action_spec = UnboundedContinuousTensorSpec(
            device=device, dtype=torch.float32, shape=torch.Size([1])
        )
        self.observation_spec = UnboundedContinuousTensorSpec(
            3, device=device, dtype=torch.float32
        )

    def reset(self, *, config: dict[str, Any] | None = None) -> torch.Tensor:
        config = config or {}
        self.max_speed = config.get("max_speed", 8)
        self.max_torque = config.get("max_torque", 2.0)
        self.dt = config.get("dt", 0.05)
        self.g = config.get("g", 10.0)
        self.m = config.get("m", 1.0)
        self.l = config.get("l", 1.0)

        th = torch.empty(
            1, self.num_envs, device=self.device, dtype=torch.float32
        ).uniform_(-torch.pi, torch.pi)
        thdot = torch.empty(
            1, self.num_envs, device=self.device, dtype=torch.float32
        ).uniform_(-1.0, 1.0)
        self.state = torch.vstack((th, thdot))
        obs = torch.vstack((torch.cos(th), torch.sin(th), thdot))
        return obs.T

    def step(self, action: torch.Tensor) -> TensorDict:
        th, thdot = self.state

        u = torch.clip(action.flatten(), -self.max_torque, self.max_torque)
        costs = (
            (((th + torch.pi) % (2 * torch.pi)) - torch.pi) ** 2
            + 0.1 * thdot**2
            + 0.001 * (u**2)
        )

        newthdot = (
            thdot
            + (
                3 * self.g / (2 * self.l) * torch.sin(th)
                + 3.0 / (self.m * self.l**2) * u
            )
            * self.dt
        )
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        self.state = torch.vstack((newth, newthdot))

        obs = torch.vstack((torch.cos(newth), torch.sin(newth), newthdot))
        return TensorDict(
            {
                DataKeys.OBS: obs.T,
                DataKeys.REWARDS: -costs.reshape(self.num_envs, 1),
            },
            batch_size=self.num_envs,
            device=self.device,
        )
