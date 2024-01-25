from dataclasses import asdict, dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec

from rl8 import Env
from rl8.data import DataKeys, Device


@torch.compile
def step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    action: torch.Tensor,
    *,
    force_mag: float = 0.001,
    goal_position: float = 0.5,
    goal_velocity: float = 0.0,
    gravity: float = 0.0025,
    max_position: float = 0.6,
    max_speed: float = 0.07,
    min_position: float = -1.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled version of the environment step for extra speed."""
    velocity += (action.flatten() - 1) * force_mag - gravity * torch.cos(3 * position)
    velocity = velocity.clip_(-max_speed, max_speed)
    position += velocity
    position = position.clip_(min_position, max_position)
    velocity[(position == min_position) & (velocity < 0)] = 0.0

    reward = (position - goal_position).abs_()
    reward *= -1
    reward[(position >= goal_position) & (velocity >= goal_velocity)] = 1.0
    state = torch.vstack((position, velocity))
    obs = state.T
    return state, obs, reward


@dataclass
class MountainCarConfig:
    # Force applied to the car.
    force_mag: float = 0.001

    # Car must be at or past this position for max reward.
    goal_position: float = 0.5

    # Car must be moving at least this fast past the position for max reward.
    goal_velocity: float = 0.0

    # Gravity pulling the car down the hill.
    gravity: float = 0.0025

    # Car max position.
    max_position: float = 0.6

    # Car max speed.
    max_speed: float = 0.07

    # Car min position.
    min_position: float = -1.2


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
        self._config = MountainCarConfig(**self.config)

    def reset(self, *, config: dict[str, Any] | None = None) -> torch.Tensor:
        config = config or {}
        self._config = MountainCarConfig(**config)
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
        self.state, obs, reward = step(
            position, velocity, action, **asdict(self._config)
        )
        return TensorDict(
            {
                DataKeys.OBS: obs,
                DataKeys.REWARDS: reward.reshape(self.num_envs, 1),
            },
            batch_size=self.num_envs,
            device=self.device,
        )
