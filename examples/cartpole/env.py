from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec

from rl8 import Env
from rl8.data import DataKeys, Device


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

        self.gravity = self.config.get("gravity", 9.8)
        self.cart_mass = self.config.get("cart_mass", 1.0)
        self.pole_mass = self.config.get("pole_mass", 0.1)
        self.total_mass = self.pole_mass + self.cart_mass
        self.length = self.config.get("length", 0.5)  # actually half the pole's length
        self.pole_mass_length = self.pole_mass * self.length
        self.force_mag = self.config.get("force_mag", 5.0)
        self.tau = self.config.get("tau", 0.02)  # seconds between state updates
        self.kinematics_integrator = self.config.get("kinematics_integrator", "euler")

    def reset(self, *, config: dict[str, Any] | None = None) -> torch.Tensor:
        config = config or {}
        self.gravity = config.get("gravity", 9.8)
        self.cart_mass = config.get("cart_mass", 1.0)
        self.pole_mass = config.get("pole_mass", 0.1)
        self.total_mass = self.pole_mass + self.cart_mass
        self.length = config.get("length", 0.5)  # actually half the pole's length
        self.pole_mass_length = self.pole_mass * self.length
        self.force_mag = config.get("force_mag", 5.0)
        self.tau = config.get("tau", 0.02)  # seconds between state updates
        self.kinematics_integrator = config.get("kinematics_integrator", "euler")
        self.state = torch.normal(
            0, 0.01, size=(4, self.num_envs), device=self.device, dtype=torch.float32
        )
        x, x_dot, theta, theta_dot = self.state
        obs = torch.vstack((x, x_dot, torch.cos(theta), torch.sin(theta), theta_dot))
        return obs.T

    def step(self, action: torch.Tensor) -> TensorDict:
        x, x_dot, theta, theta_dot = self.state
        force = (action.flatten() - 1) * self.force_mag
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        tmp = (
            force + self.pole_mass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * tmp) / (
            self.length * (4.0 / 3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        x_acc = tmp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_acc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * theta_acc
            theta = theta + self.tau * theta_dot

        self.state = torch.vstack((x, x_dot, theta, theta_dot))
        obs = torch.vstack((x, x_dot, torch.cos(theta), torch.sin(theta), theta_dot))
        theta_vector = obs[2:-1, :]
        theta_ref = torch.zeros_like(theta_vector)
        theta_ref[0, :] = 1.0
        theta_error = torch.abs(theta_vector - theta_ref).sum(axis=0, keepdim=True).T
        other_errors = (
            torch.abs(torch.vstack((self.state[0], self.state[1], self.state[-1])))
            .sum(axis=0, keepdim=True)
            .T
        )
        reward = theta_error + other_errors
        return TensorDict(
            {
                DataKeys.OBS: obs.T,
                DataKeys.REWARDS: -reward,
            },
            batch_size=self.num_envs,
            device=self.device,
        )
