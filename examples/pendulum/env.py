from dataclasses import asdict, dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec

from rl8 import Env
from rl8.data import DataKeys, Device


@torch.compile
def step(
    th: torch.Tensor,
    thdot: torch.Tensor,
    action: torch.Tensor,
    *,
    dt: float = 0.05,
    g: float = 10.0,
    l: float = 1.0,
    m: float = 1.0,
    max_speed: float = 8.0,
    max_torque: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled version of the environment step for extra speed."""
    u = torch.clip(action.flatten(), -max_torque, max_torque)
    costs = (
        (((th + torch.pi) % (2 * torch.pi)) - torch.pi) ** 2
        + 0.1 * thdot**2
        + 0.001 * (u**2)
    )

    newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
    newthdot = newthdot.clip_(-max_speed, max_speed)
    newth = th + newthdot * dt

    state = torch.vstack((newth, newthdot))
    obs = torch.vstack((torch.cos(newth), torch.sin(newth), newthdot)).T
    return state, obs, -costs


@dataclass
class PendulumConfig:
    # Timestep between step calls.
    dt: float = 0.05

    # Gravity.
    g: float = 10.0

    # Pendulum length.
    l: float = 1.0

    # System mass.
    m: float = 1.0

    # Pendulum max angular speed.
    max_speed: float = 8.0

    # Max torque that can be applied to the pendulum.
    max_torque: float = 2.0


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
        self._config = PendulumConfig(**self.config)

        self.action_spec = UnboundedContinuousTensorSpec(
            device=device, dtype=torch.float32, shape=torch.Size([1])
        )
        self.observation_spec = UnboundedContinuousTensorSpec(
            3, device=device, dtype=torch.float32
        )

    def reset(self, *, config: dict[str, Any] | None = None) -> torch.Tensor:
        config = config or {}
        self._config = PendulumConfig(**config)

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
        self.state, obs, reward = step(th, thdot, action, **asdict(self._config))
        return TensorDict(
            {
                DataKeys.OBS: obs,
                DataKeys.REWARDS: reward.reshape(self.num_envs, 1),
            },
            batch_size=self.num_envs,
            device=self.device,
        )
