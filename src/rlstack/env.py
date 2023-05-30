"""Environment protocol definition and helper dummy environment definitions."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch
from tensordict import TensorDict

from .data import DataKeys, Device
from .specs import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec

_ObservationSpec = TypeVar("_ObservationSpec", bound=TensorSpec)
_ActionSpec = TypeVar("_ActionSpec", bound=TensorSpec)


class Env(ABC):
    """Protocol defining the IsaacGym -like environments for supporting
    highly parallelized simulation.

    Args:
        num_envs: Number of parallel and independent environments being
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

    #: Environment config passed to the environment at instantiation.
    #: This could be overwritten by :meth:`Environment.reset`, but it's
    #: entirely at the developer's discretion.
    config: dict[str, Any]

    #: Device the environment's states, observations, and rewards reside
    #: on.
    device: Device

    #: An optional attribute denoting the max number of steps an environment
    #: may take before being reset.
    max_horizon: int

    #: Number of parallel and independent environments being simulated.
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
        self.num_envs = num_envs
        self.config = config or {}
        self.device = device

    @abstractmethod
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

    @abstractmethod
    def step(self, action: torch.Tensor | TensorDict) -> TensorDict:
        """Step the environment by applying an action, simulating an environment
        transition, and returning an observation and a reward.

        Args:
            action: Action to apply to the environment with tensor spec
                :attr:`Env.action_spec`.

        Returns:
            A tensordict containing "obs" and "rewards" keys and values.

        """


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
        super().__init__(num_envs, config=config, device=device)
        self.observation_spec = UnboundedContinuousTensorSpec(1, device=self.device)
        self.bounds = self.config.get("bounds", 100.0)

    def reset(self, *, config: None | dict[str, Any] = None) -> torch.Tensor:
        config = config or {}
        self.bounds = config.get("bounds", self.bounds)
        self.state = torch.empty(self.num_envs, 1, device=self.device).uniform_(
            -self.bounds, self.bounds
        )
        return self.state


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
