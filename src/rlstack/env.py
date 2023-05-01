"""Environment protocol definition and helper dummy environment definitions."""

from typing import Any, Generic, Protocol, TypeVar

import torch
from tensordict import TensorDict
from typing_extensions import Self

from .data import DataKeys, Device
from .specs import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec

_ObservationSpec = TypeVar("_ObservationSpec", bound=TensorSpec)
_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)


class Env(Protocol):
    """Protocol defining the IsaacGym -like and OpenAI Gym -like environment
    for supporting highly parallelized simulation.

    Args:
        num_envs: Number of parallel and independent environment being
            simulated by one :class:`Env` instance.
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
    device: Device

    #: Max number of steps an environment may take before being reset.
    max_horizon: int

    #: Number of parallel and independent environments being simulated
    #: by one :class:`Env` instance. If the learning buffer has batch size
    #: ``[B, T]``, ``num_envs`` would be equivalent to ``B``.
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
        device: Device = "cpu",
    ) -> None:
        ...

    def reset(
        self, *, config: None | dict[str, Any] = None
    ) -> torch.Tensor | TensorDict:
        """Reset the environment, applying a new environment config to it and
        returning a new, initial observation from the environment.

        Args:
            config: Environment configuration/options/parameters.

        Returns:
            Initial observation from the reset environment with spec
            :attr:`Env.observation_spec`.

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

    def to(self, device: Device, /) -> Self:
        """Move the environment and its attributes to `device`."""


class GenericEnv(Env, Generic[_ObservationSpec, _ActionSpec]):
    """Generic version of `Env` for environments with constant specs."""

    #: Environment observation spec.
    observation_spec: _ObservationSpec

    #: Environment aciton spec.
    action_spec: _ActionSpec


class DummyEnv(GenericEnv[UnboundedContinuousTensorSpec, _ActionSpec]):
    """The simplest environment possible.

    Useful for testing and debugging algorithms and policies. The state
    is just a position along a 1D axis and the action perturbs the
    state by some amount. The reward is the negative of the state's distance
    from the origin, incentivizing policies to drive the state to the
    origin as quickly as possible.

    The environment's action space and step functions are defined by
    subclasses.

    """

    #: State magnitude bounds for generating initial states upon
    #: environment creation and environment resets.
    bounds: float

    #: Current environment state that's a position along a 1D axis.
    state: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        /,
        *,
        config: None | dict[str, Any] = None,
        device: Device = "cpu",
    ) -> None:
        if config is None:
            config = {}
        self.num_envs = num_envs
        self.observation_spec = UnboundedContinuousTensorSpec(1, device=device)
        self.bounds = config.get("bounds", 100.0)
        self.state = (
            -self.bounds * torch.rand(num_envs, device=device).unsqueeze(1)
            + self.bounds
        )
        self.device = device

    def reset(self, *, config: None | dict[str, Any] = None) -> torch.Tensor:
        if config and "bounds" in config:
            self.bounds = config["bounds"]
        num_envs = self.state.size(0)
        self.state = (
            -self.bounds * torch.rand(num_envs, device=self.device).unsqueeze(1)
            + self.bounds
        )
        return self.state

    def to(self, device: Device, /) -> Self:
        self.observation_spec = self.observation_spec.to(device)  # type: ignore[assignment]
        self.state = self.state.to(device=device)
        self.device = device
        return self


class ContinuousDummyEnv(DummyEnv[UnboundedContinuousTensorSpec]):
    """A continuous version of the dummy environment.

    Actions include moving the state left or right at any magnitude.

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
        self.action_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([1]), device=device
        )

    def step(self, action: torch.Tensor) -> TensorDict:
        self.state += action
        return TensorDict(
            {DataKeys.OBS: self.state, DataKeys.REWARDS: -self.state.abs()},
            batch_size=self.num_envs,
            device=self.device,
        )


class DiscreteDummyEnv(DummyEnv[DiscreteTensorSpec]):
    """A discrete version of the dummy environment.

    Actions include moving the state left or right one unit. This
    environment is considered more difficult to solve than its
    continuous counterpart because of the limited action space.

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
        self.action_spec = DiscreteTensorSpec(2, shape=torch.Size([1]), device=device)

    def step(self, action: torch.Tensor) -> TensorDict:
        self.state += 2 * action - 1
        return TensorDict(
            {DataKeys.OBS: self.state, DataKeys.REWARDS: -self.state.abs()},
            batch_size=self.num_envs,
            device=self.device,
        )
