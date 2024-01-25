from dataclasses import asdict, dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec

from rl8 import Env
from rl8.data import DataKeys, Device


@torch.compile
def step(
    x: torch.Tensor,
    x_dot: torch.Tensor,
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    action: torch.Tensor,
    *,
    force_mag: float = 5.0,
    gravity: float = 9.8,
    kinematics_integrator: str = "euler",
    length: float = 0.5,
    pole_mass: float = 0.1,
    pole_mass_length: float = 0.05,
    total_mass: float = 1.1,
    tau: float = 0.02,
    **_,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled version of the environment step for extra speed."""
    force = (action.flatten() - 1) * force_mag
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    tmp = (force + pole_mass_length * theta_dot**2 * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * tmp) / (
        length * (4.0 / 3.0 - pole_mass * costheta**2 / total_mass)
    )
    x_acc = tmp - pole_mass_length * theta_acc * costheta / total_mass

    if kinematics_integrator == "euler":
        x = x + tau * x_dot
        x_dot = x_dot + tau * x_acc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * theta_acc
    else:  # semi-implicit euler
        x_dot = x_dot + tau * x_acc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * theta_acc
        theta = theta + tau * theta_dot

    state = torch.vstack((x, x_dot, theta, theta_dot))
    obs = torch.vstack((x, x_dot, torch.cos(theta), torch.sin(theta), theta_dot))
    theta_vector = obs[2:-1, :]
    theta_ref = torch.zeros_like(theta_vector)
    theta_ref[0, :] = 1.0
    theta_error = (theta_vector - theta_ref).abs_().sum(axis=0, keepdim=True).T
    other_errors = (
        torch.vstack((state[0], state[1], state[-1])).abs_().sum(axis=0, keepdim=True).T
    )
    reward = theta_error + other_errors
    return state, obs.T, -reward


@dataclass
class CartPoleConfig:
    # Cart mass.
    cart_mass: float = 1.0

    # Force magnitude applied to the cart.
    force_mag: float = 5.0

    # Gravity.
    gravity: float = 9.8

    # Integrator.
    kinematics_integrator: str = "euler"

    # Pole length.
    length: float = 0.5

    # Pole mass.
    pole_mass: float = 0.1

    # Pole mass * pole length. Overwritten later.
    pole_mass_length: float = 0.05

    # Pole mass + cart mass. Overwritten later.
    total_mass: float = 1.1

    # Timestep.
    tau: float = 0.02

    def __post_init__(self) -> None:
        self.pole_mass_length = self.pole_mass * self.length
        self.total_mass = self.cart_mass + self.pole_mass


class CartPole(Env):
    """Reimplementation of the classic `CartPole`_ environment.

    .. _`CartPole`: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py

    """

    # Max number of steps that can be taken with the environment before
    # requiring a reset.
    max_horizon = 128

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
            5, device=device, dtype=torch.float32
        )
        self.action_spec = DiscreteTensorSpec(3, shape=torch.Size([1]), device=device)
        self._config = CartPoleConfig(**self.config)

    def reset(self, *, config: dict[str, Any] | None = None) -> torch.Tensor:
        config = config or {}
        self._config = CartPoleConfig(**config)
        self.state = torch.normal(
            0, 0.01, size=(4, self.num_envs), device=self.device, dtype=torch.float32
        )
        x, x_dot, theta, theta_dot = self.state
        obs = torch.vstack((x, x_dot, torch.cos(theta), torch.sin(theta), theta_dot))
        return obs.T

    def step(self, action: torch.Tensor) -> TensorDict:
        x, x_dot, theta, theta_dot = self.state
        self.state, obs, reward = step(
            x, x_dot, theta, theta_dot, action, **asdict(self._config)
        )
        return TensorDict(
            {
                DataKeys.OBS: obs,
                DataKeys.REWARDS: reward,
            },
            batch_size=self.num_envs,
            device=self.device,
        )
