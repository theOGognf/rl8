from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec

from rl8 import Env
from rl8.data import DataKeys, Device


class MountainCar(Env):
    """Reimplementation of the classic `MountainCar`_ environment.

    .. _`MountainCar`: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py

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
        self.observation_spec = UnboundedContinuousTensorSpec(
            2, device=device, dtype=torch.float32
        )
        self.action_spec = DiscreteTensorSpec(3, shape=torch.Size([1]), device=device)

        self.min_position = self.config.get("min_position", -1.2)
        self.max_position = self.config.get("pole_mass", 0.6)
        self.max_speed = self.config.get("max_speed", 0.07)
        self.goal_position = self.config.get("goal_position", 0.5)
        self.goal_velocity = self.config.get("goal_velocity", 0.0)
        self.gravity = self.config.get("gravity", 0.0025)
        self.force_mag = self.config.get("force_mag", 0.001)

    def reset(self, *, config: dict[str, Any] | None = None) -> torch.Tensor:
        config = config or {}
        self.min_position = config.get("min_position", -1.2)
        self.max_position = config.get("pole_mass", 0.6)
        self.max_speed = config.get("max_speed", 0.07)
        self.goal_position = config.get("goal_position", 0.5)
        self.goal_velocity = config.get("goal_velocity", 0.0)
        self.gravity = config.get("gravity", 0.0025)
        self.force_mag = config.get("force_mag", 0.001)
        position = torch.normal(
            -0.5, 0.05, size=(1, self.num_envs), device=self.device, dtype=torch.float32
        )
        velocity = torch.normal(
            0, 0.05, size=(1, self.num_envs), device=self.device, dtype=torch.float32
        )
        self.state = torch.vstack((position, velocity))
        return self.state.T

    def step(self, action: torch.Tensor) -> TensorDict:
        position, velocity = self.state
        velocity += (action.flatten() - 1) * self.force_mag - self.gravity * torch.cos(
            3 * position
        )
        velocity = torch.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = torch.clip(position, self.min_position, self.max_position)
        velocity[(position == self.min_position) & (velocity < 0)] = 0.0

        reward = -torch.abs(position - self.goal_position)
        reward[
            (position >= self.goal_position) & (velocity >= self.goal_velocity)
        ] = 1.0

        self.state = torch.vstack((position, velocity))
        return TensorDict(
            {
                DataKeys.OBS: self.state.T,
                DataKeys.REWARDS: reward.reshape(self.num_envs, 1),
            },
            batch_size=self.num_envs,
            device=self.device,
        )
