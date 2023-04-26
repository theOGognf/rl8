"""Environment protocol definition and helper dummy environment definitions."""

from typing import Any, Generic, Protocol, TypeVar

import torch
from tensordict import TensorDict
from typing_extensions import Self

from ..specs import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec
from .data import DEVICE, DataKeys

_ObservationSpec = TypeVar("_ObservationSpec", bound=TensorSpec)
_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)


class Env(Protocol):
    """Protocol defining the IsaacGym -like and OpenAI Gym -like environment
    for supporting highly parallelized simulation.

    Args:
        num_envs: Number of parallel and independent environment being
            simulated by one `Env` instance.
        config: Config detailing simulation options/parameters for the
            environment's initialization.
        device: Device the environment's underlying data should be
            initialized on.

    """

    #: Spec defining the environment's inputs (and policy's action
    #: distribution's outputs). Used for initializing the policy, the
    #: policy's underlying components, and the learning buffer.
    action_spec: TensorSpec

    #: Current environment config detailing simulation options or
    #: parameters.
    config: dict[str, Any]

    #: Device the environment's underlying data is on.
    device: DEVICE

    #: Max number of steps an environment may take before being reset.
    max_horizon: int

    #: Number of parallel and independent environments being simulated
    #: by one `Env` instance. If the learning buffer has batch size
    #: [B, T], `num_envs` would be equivalent to B.
    num_envs: int

    #: Spec defining part of the environment's outputs (and policy's
    #: model's outputs). Used for initializing the policy, the
    #: policy's underlying components, and the learning buffer.
    observation_spec: TensorSpec

    def __init__(
        self,
        num_envs: int,
        /,
        *,
        config: None | dict[str, Any] = None,
        device: DEVICE = "cpu",
    ) -> None:
        ...

    def close(self) -> None:
        """Close the environment, releasing all resources (memory, files, etc.)
        associated with it. This is typically only called once at the end of
        learning, so it's usually not useful. `reset` should handle most of
        the resource allocation/freeing for the most part.

        """

    def reset(
        self, *, config: None | dict[str, Any] = None
    ) -> torch.Tensor | TensorDict:
        """Reset the environment, applying a new environment config to it and
        returning a new, initial observation from the environment.

        Args:
            config: Environment configuration/options/parameters.

        Returns:
            Initial observation from the reset environment with spec
            `Env.observation_spec`.

        """

    def step(self, action: torch.Tensor | TensorDict) -> TensorDict:
        """Step the environment by applying an action, simulating an environment
        transition, and returning an observation and a reward.

        Args:
            action: Action to apply to the environment with tensor spec
                `Env.action_spec`.

        Returns:
            A tensordict containing keys and values:
                - "obs": New environment observations.
                - "rewards": New environment rewards.

        """

    def to(self, device: DEVICE, /) -> Self:
        """Move the environment and its attributes to `device`."""


class GenericEnv(Env, Generic[_ObservationSpec, _ActionSpec]):
    """Generic version of `Env` for environments with constant specs."""


class DummyEnv(GenericEnv[UnboundedContinuousTensorSpec, _ActionSpec]):

    bounds: float

    state: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        /,
        *,
        config: None | dict[str, Any] = None,
        device: DEVICE = "cpu",
    ) -> None:
        if config is None:
            config = {}
        self.num_envs = num_envs
        self.observation_spec = UnboundedContinuousTensorSpec(1, device=device)
        self.bounds = config.get("bounds", 100.0)
        self.state = self.bounds * torch.rand(num_envs, device=device).unsqueeze(1)

    def reset(self, *, config: None | dict[str, Any] = None) -> torch.Tensor:
        if config and "bounds" in config:
            self.bounds = config["bounds"]
        num_envs = self.state.size(0)
        device = self.state.device
        self.state = self.bounds * torch.rand(num_envs, device=device).unsqueeze(1)
        return self.state

    def to(self, device: DEVICE, /) -> Self:
        self.state = self.state.to(device=device)
        return self


class ContinuousDummyEnv(DummyEnv[UnboundedContinuousTensorSpec]):
    def __init__(
        self,
        num_envs: int,
        /,
        *,
        config: dict[str, Any] | None = None,
        device: DEVICE = "cpu",
    ) -> None:
        super().__init__(num_envs, config=config, device=device)
        self.action_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([1]), device=device
        )

    def step(self, action: torch.Tensor) -> TensorDict:
        self.state += action
        return TensorDict(
            {DataKeys.OBS: self.state, DataKeys.REWARDS: -self.state.abs()},
            batch_size=self.num_envs,
        )


class DiscreteDummyEnv(DummyEnv[DiscreteTensorSpec]):
    def __init__(
        self,
        num_envs: int,
        /,
        *,
        config: dict[str, Any] | None = None,
        device: DEVICE = "cpu",
    ) -> None:
        super().__init__(num_envs, config=config, device=device)
        self.action_spec = DiscreteTensorSpec(2, shape=torch.Size([1]), device=device)

    def step(self, action: torch.Tensor) -> TensorDict:
        self.state += 2 * action - 1
        return TensorDict(
            {DataKeys.OBS: self.state, DataKeys.REWARDS: -self.state.abs()},
            batch_size=self.num_envs,
        )
