"""Definitions related to a feedforward PPO algorithm (specifically focused
on data collection and training on many environments in parallel).

"""

from dataclasses import asdict
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dadaptation import DAdaptAdam
from tensordict import TensorDict
from torch.utils.data import DataLoader
from typing_extensions import Self

from .._utils import profile_ms
from ..data import (
    AlgorithmHparams,
    AlgorithmState,
    CollectStats,
    DataKeys,
    Device,
    StepStats,
)
from ..distributions import Distribution
from ..env import Env
from ..models import Model
from ..policies import Policy
from ..schedulers import EntropyScheduler, LRScheduler, ScheduleKind
from ..specs import CompositeSpec, UnboundedContinuousTensorSpec


class Algorithm:
    """An optimized feedforward `PPO`_ algorithm with common tricks for
    stabilizing and accelerating learning.

    This algorithm assumes environments are parallelized much like
    `IsaacGym environments`_ and are infinite horizon with no terminal
    conditions. These assumptions allow the learning procedure to occur
    extremely fast even for complex, sequence-based models because:

        - Environments occur in parallel and are batched into a contingous
          buffer.
        - All environments are reset in parallel after a predetermined
          horizon is reached.
        - All operations occur on the same device, removing overhead
          associated with data transfers between devices.

    Args:
        env_cls: Highly parallelized environment for sampling experiences.
            Instantiated with ``env_config``. Will be stepped for ``horizon``
            each :meth:`Algorithm.collect` call.
        env_config: Initial environment config passed to ``env_cls`` for
            environment instantiation. This is likely to be overwritten
            on the environment instance if reset with a new config.
        model: Model instance to use. Mutually exclusive with ``model_cls``.
        model_cls: Optional custom policy model definition. A model class
            is provided for you based on the environment instance's specs
            if you don't provide one. Defaults to a simple feedforward
            neural network.
        model_config: Optional policy model config unpacked into the model
            during instantiation.
        dist_cls: Custom policy action distribution class. An action
            distribution class is provided for you based on the environment
            instance's specs if you don't provide one. Defaults to a categorical
            action distribution for discrete actions and a normal action
            distribution for continuous actions. Complex actions are not
            supported for default action distributions.
        horizon: Number of environment transitions to collect during
            :meth:`Algorithm.collect`. The environment is reset based on
            ``horizons_per_env_reset``. The buffer's size is ``[B, T]`` where ``T`` is
            ``horizon``.
        horizons_per_env_reset: Number of times :meth:`Algorithm.collect` can be
            called before resetting :attr:`Algorithm.env`. Set this to a higher
            number if you want learning to occur across horizons. Leave this
            as the default ``1`` if it doesn't matter that experiences and
            learning only occurs within one horizon.
        num_envs: Number of parallelized simulation environments for the
            environment instance. Passed during the environment's
            instantiation. The buffer's size is ``[B, T]`` where ``B`` is
            ``num_envs``.
        optimizer_cls: Custom optimizer class. Defaults to an optimizer
            that doesn't require much tuning.
        optimizer_config: Custom optimizer config unpacked into ``optimizer_cls``
            during optimizer instantiation.
        lr_schedule: Optional schedule that overrides the optimizer's learning rate.
            This deternmines the value of the learning rate according to the
            number of environment transitions experienced during learning.
            The learning rate is constant if this isn't provided.
        lr_schedule_kind: Kind of learning rate scheduler to use if ``lr_schedule``
            is provided. Options include:

                - "step": jump to values and hold until a new environment transition
                  count is reached.
                - "interp": jump to values like "step", but interpolate between the
                  current value and the next value.

        entropy_coeff: Entropy coefficient value. Weight of the entropy loss w.r.t.
            other components of total loss. This value is ignored if
            ``entropy_coeff_schedule`` is provded.
        entropy_coeff_schedule: Optional schedule that overrides ``entropy_coeff``. This
            determines values of ``entropy_coeff`` according to the number of environment
            transitions experienced during learning.
        entropy_coeff_schedule_kind: Kind of entropy scheduler to use. Options include:

            - "step": jump to values and hold until a new environment transition
              count is reached.
            - "interp": jump to values like "step", but interpolate between the
              current value and the next value.

        gae_lambda: Generalized Advantage Estimation (GAE) hyperparameter for controlling
            the variance and bias tradeoff when estimating the state value
            function from collected environment transitions. A higher value
            allows higher variance while a lower value allows higher bias
            estimation but lower variance.
        gamma: Discount reward factor often used in the Bellman operator for
            controlling the variance and bias tradeoff in collected experienced
            rewards. Note, this does not control the bias/variance of the
            state value estimation and only controls the weight future rewards
            have on the total discounted return.
        sgd_minibatch_size: PPO hyperparameter indicating the minibatch size
            :attr:`Algorithm.buffer` is split into when updating the policy's model
            in :meth:`Algorithm.step`. It's usually best to maximize the minibatch
            size to reduce the variance associated with updating the policy's model,
            and also to accelerate the computations when learning (assuming a CUDA
            device is being used). If ``None``, the whole buffer is treated as one giant
            batch.
        num_sgd_iter: PPO hyperparameter indicating the number of gradient steps to take
            with the whole :attr:`Algorithm.buffer` when calling :meth:`Algorithm.step`.
        shuffle_minibatches: Whether to shuffle minibatches within :meth:`Algorithm.step`.
            Recommended, but not necessary if the minibatch size is large enough
            (e.g., the buffer is the batch).
        clip_param: PPO hyperparameter indicating the max distance the policy can
            update away from previously collected policy sample data with
            respect to likelihoods of taking actions conditioned on
            observations. This is the main innovation of PPO.
        vf_clip_param: PPO hyperparameter similar to ``clip_param`` but for the
            value function estimate. A measure of max distance the model's
            value function is allowed to update away from previous value function
            samples.
        dual_clip_param: PPO hyperparameter that clips like ``clip_param`` but when
            advantage estimations are negative. Helps prevent instability for
            continuous action spaces when policies are making large updates.
            Leave ``None`` for this clip to not apply. Otherwise, typical values
            are around ``5``.
        vf_coeff: Value function loss component weight. Only needs to be tuned
            when the policy and value function share parameters.
        max_grad_norm: Max gradient norm allowed when updating the policy's model
            within :meth:`Algorithm.step`.
        device: Device :attr:`Algorithm.env`, :attr:`Algorithm.buffer`, and
            :attr:`Algorithm.policy` all reside on.

    Examples:
        Instantiate an algorithm for a dummy environment and update underlying
        policy once.

        >>> from rlstack import Algorithm
        >>> from rlstack.env import DiscreteDummyEnv
        >>> algo = Algorithm(DiscreteDummyEnv)
        >>> algo.collect()
        >>> algo.step()  # doctest: +SKIP

    .. _`PPO`: https://arxiv.org/pdf/1707.06347.pdf
    .. _`IsaacGym environments`: https://arxiv.org/pdf/2108.10470.pdf

    """

    #: Environment experience buffer used for aggregating environment
    #: transition data and policy sample data. The same buffer object
    #: is shared whenever using :meth:`Algorithm.collect` Buffer dimensions
    #: are determined by ``num_envs`` and ``horizon`` args.
    buffer: TensorDict

    #: Tensor spec defining the environment experience buffer components
    #: and dimensions. Used for instantiating :attr:`Algorithm.buffer`
    #: at :class:`Algorithm` instantiation and each :meth:`Algorithm.step`
    #: call.
    buffer_spec: CompositeSpec

    #: Entropy scheduler for updating the ``entropy_coeff`` after each
    #: :meth:`Algorithm.step` call based on the number environment transitions
    #: collected and learned on. By default, the entropy scheduler does not
    #: actually update the entropy coefficient. The entropy scheduler only
    #: updates the entropy coefficient if an ``entropy_coeff_schedule`` is
    #: provided.
    entropy_scheduler: EntropyScheduler

    #: Environment used for experience collection within the
    #: :meth:`Algorithm.collect` method. It's ultimately up to the environment
    #: to make learning efficient by parallelizing simulations.
    env: Env

    #: Feedforward PPO hyperparameters that're constant throughout training
    #: and can drastically affect training performance.
    hparams: AlgorithmHparams

    #: Learning rate scheduler for updating `optimizer` learning rate after
    #: each `step` call based on the number of environment transitions
    #: collected and learned on. By default, the learning scheduler does not
    #: actually alter the `optimizer` learning rate (it actually leaves it
    #: constant). The learning rate scheduler only alters the learning rate
    #: if a `learning_rate_schedule` is provided.
    lr_scheduler: LRScheduler

    #: Underlying optimizer for updating the policy's model. Constructed from
    #: ``optimizer_cls`` and ``optimizer_config``. Defaults to a generally
    #: robust optimizer that doesn't require much hyperparameter tuning.
    optimizer: optim.Optimizer

    #: Policy constructed from the ``model_cls``, ``model_config``, and
    #: ``dist_cls`` kwargs. A default policy is constructed according to
    #: the environment's observation and action specs if these policy args
    #: aren't provided. The policy is what does all the action sampling
    #: within :meth:`Algorithm.collect` and is what is updated within
    #: :meth:`Algorithm.step`.
    policy: Policy

    #: Algorithm state for determining when to reset the environment, when
    #: the policy can be updated, and for tracking additional algorithm
    #: metrics like method call counts.
    state: AlgorithmState

    def __init__(
        self,
        env_cls: type[Env],
        /,
        *,
        env_config: None | dict[str, Any] = None,
        model: None | Model = None,
        model_cls: None | type[Model] = None,
        model_config: None | dict[str, Any] = None,
        dist_cls: None | type[Distribution] = None,
        horizon: None | int = 32,
        horizons_per_env_reset: int = 1,
        num_envs: int = 8192,
        optimizer_cls: type[optim.Optimizer] = DAdaptAdam,
        optimizer_config: None | dict[str, Any] = None,
        lr_schedule: None | list[tuple[int, float]] = None,
        lr_schedule_kind: ScheduleKind = "step",
        entropy_coeff: float = 0.0,
        entropy_coeff_schedule: None | list[tuple[int, float]] = None,
        entropy_coeff_schedule_kind: ScheduleKind = "step",
        gae_lambda: float = 0.95,
        gamma: float = 0.95,
        sgd_minibatch_size: None | int = None,
        num_sgd_iter: int = 4,
        shuffle_minibatches: bool = True,
        clip_param: float = 0.2,
        vf_clip_param: float = 5.0,
        dual_clip_param: None | float = None,
        vf_coeff: float = 1.0,
        max_grad_norm: float = 5.0,
        device: Device = "cpu",
    ) -> None:
        self.env = env_cls(num_envs, config=env_config, device=device)
        self.policy = Policy(
            self.env.observation_spec,
            self.env.action_spec,
            model=model,
            model_cls=model_cls,
            model_config=model_config,
            dist_cls=dist_cls,
            device=device,
        )
        max_horizon = self.env.max_horizon if hasattr(self.env, "max_horizon") else 32
        horizon = min(horizon, max_horizon) if horizon else max_horizon
        self.buffer_spec = CompositeSpec(  # type: ignore[no-untyped-call]
            {
                DataKeys.OBS: self.env.observation_spec,
                DataKeys.REWARDS: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.FEATURES: self.policy.feature_spec,
                DataKeys.ACTIONS: self.env.action_spec,
                DataKeys.LOGP: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.VALUES: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.ADVANTAGES: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.RETURNS: UnboundedContinuousTensorSpec(1, device=device),
            },
        ).to(device)
        self.buffer = self.buffer_spec.zero([num_envs, horizon + 1])
        optimizer_config = optimizer_config or {}
        self.optimizer = optimizer_cls(
            self.policy.model.parameters(), **optimizer_config
        )
        self.lr_scheduler = LRScheduler(
            self.optimizer, schedule=lr_schedule, kind=lr_schedule_kind
        )
        self.entropy_scheduler = EntropyScheduler(
            entropy_coeff,
            schedule=entropy_coeff_schedule,
            kind=entropy_coeff_schedule_kind,
        )
        sgd_minibatch_size = (
            sgd_minibatch_size if sgd_minibatch_size else num_envs * horizon
        )
        self.hparams = AlgorithmHparams(
            clip_param=clip_param,
            dual_clip_param=dual_clip_param,
            gae_lambda=gae_lambda,
            gamma=gamma,
            horizon=horizon,
            horizons_per_env_reset=horizons_per_env_reset,
            max_grad_norm=max_grad_norm,
            num_sgd_iter=num_sgd_iter,
            sgd_minibatch_size=sgd_minibatch_size,
            shuffle_minibatches=shuffle_minibatches,
            vf_clip_param=vf_clip_param,
            vf_coeff=vf_coeff,
        )
        self.state = AlgorithmState()

    def collect(
        self, *, env_config: None | dict[str, Any] = None, deterministic: bool = False
    ) -> CollectStats:
        """Collect environment transitions and policy samples in a buffer.

        This is one of the main :class:`Algorithm` methods. This is usually
        called immediately prior to :meth:`Algorithm.step` to collect
        experiences used for learning.

        The environment is reset immediately prior to collecting
        transitions according to ``horizons_per_env_reset``. If
        the environment isn't reset, then the last observation is used as
        the initial observation.

        This method sets the ``buffered`` flag to enable calling
        of :meth:`Algorithm.step` so it isn't called with dummy data.

        Args:
            env_config: Optional config to pass to the environment's reset
                method. This isn't used if the environment isn't scheduled
                to be reset according to ``horizons_per_env_reset``.
            deterministic: Whether to sample from the policy deterministically.
                This is usally ``False`` during learning and ``True`` during
                evaluation.

        Returns:
            Summary statistics related to the collected experiences and
            policy samples.

        """
        with profile_ms() as collect_timer:
            # Get number of environments and horizon. Remember, there's an extra
            # sample in the horizon because we store the final environment observation
            # for the next :meth:`Algorithm.collect` call and value function estimates
            # for bootstrapping.
            B = self.num_envs
            T = self.horizon

            # Gather initial observation.
            if not (self.state.horizons % self.hparams.horizons_per_env_reset):
                self.buffer[DataKeys.OBS][:, 0, ...] = self.env.reset(config=env_config)
            else:
                self.buffer[DataKeys.OBS][:, 0, ...] = self.buffer[DataKeys.OBS][
                    :, -1, ...
                ]

            for t in range(T):
                # Sample the policy and step the environment.
                in_batch = self.buffer[:, : (t + 1), ...]
                sample_batch = self.policy.sample(
                    in_batch,
                    kind="last",
                    deterministic=deterministic,
                    inplace=False,
                    requires_grad=False,
                    return_actions=True,
                    return_logp=True,
                    return_values=True,
                    return_views=False,
                )
                out_batch = self.env.step(sample_batch[DataKeys.ACTIONS])

                # Update the buffer using sampled policy data and environment
                # transition data.
                self.buffer[DataKeys.FEATURES][:, t, ...] = sample_batch[
                    DataKeys.FEATURES
                ]
                self.buffer[DataKeys.ACTIONS][:, t, ...] = sample_batch[
                    DataKeys.ACTIONS
                ]
                self.buffer[DataKeys.LOGP][:, t, ...] = sample_batch[DataKeys.LOGP]
                self.buffer[DataKeys.VALUES][:, t, ...] = sample_batch[DataKeys.VALUES]
                self.buffer[DataKeys.REWARDS][:, t, ...] = out_batch[DataKeys.REWARDS]
                self.buffer[DataKeys.OBS][:, t + 1, ...] = out_batch[DataKeys.OBS]

            # Sample features and value function at last observation.
            in_batch = self.buffer[:, :, ...]
            sample_batch = self.policy.sample(
                in_batch,
                kind="last",
                deterministic=deterministic,
                inplace=False,
                requires_grad=False,
                return_actions=False,
                return_logp=False,
                return_values=True,
                return_views=False,
            )
            self.buffer[DataKeys.FEATURES][:, -1, ...] = sample_batch[DataKeys.FEATURES]
            self.buffer[DataKeys.VALUES][:, -1, ...] = sample_batch[DataKeys.VALUES]

            self.state.horizons += 1
            self.state.buffered = True

            # Aggregate some metrics.
            rewards = self.buffer[DataKeys.REWARDS][:, :-1, ...]
            returns = torch.sum(rewards, dim=1)
            collect_stats: CollectStats = {
                "returns/min": float(torch.min(returns)),
                "returns/max": float(torch.max(returns)),
                "returns/mean": float(torch.mean(returns)),
                "returns/std": float(torch.std(returns)),
                "rewards/min": float(torch.min(rewards)),
                "rewards/max": float(torch.max(rewards)),
                "rewards/mean": float(torch.mean(rewards)),
                "rewards/std": float(torch.std(rewards)),
            }
        self.state.collect_calls += 1
        self.state.total_steps += B * T
        collect_stats["counting/collect_calls"] = self.state.collect_calls
        collect_stats["counting/horizons"] = self.state.horizons
        collect_stats["counting/total_steps"] = self.state.total_steps
        collect_stats["profiling/collect_ms"] = collect_timer()
        return collect_stats

    @property
    def device(self) -> Device:
        """Return the device the policy is residing on."""
        return self.policy.device

    @property
    def horizon(self) -> int:
        """Max number of transitions to run for each environment."""
        return int(self.buffer.size(1)) - 1

    @property
    def num_envs(self) -> int:
        """Number of environments ran in parallel."""
        return int(self.buffer.size(0))

    @property
    def params(self) -> dict[str, Any]:
        """Return algorithm parameters."""
        return {
            "env_cls": self.env.__class__.__name__,
            "model_cls": self.policy.model.__class__.__name__,
            "dist_cls": self.policy.dist_cls.__name__,
            "optimizer_cls": self.optimizer.__class__.__name__,
            "entropy_coeff": self.entropy_scheduler.coeff,
            **asdict(self.hparams),
        }

    def step(self) -> StepStats:
        """Take a step with the algorithm, using collected environment
        experiences to update the policy.

        Returns:
            Data associated with the step (losses, loss coefficients, etc.).

        """
        if not self.state.buffered:
            raise RuntimeError(
                f"{self.__class__.__name__} is not buffered. "
                "Call `collect` once prior to `step`."
            )

        with profile_ms() as step_timer:
            # Get number of environments and horizon. Remember, there's an extra
            # sample in the horizon because we store the final environment observation
            # for the next :meth:`Algorithm.collect` call and value function estimates
            # for bootstrapping.
            B = self.num_envs
            T = self.horizon

            # Generalized Advantage Estimation (GAE) and returns bootstrapping.
            prev_advantage = 0.0
            for t in reversed(range(T)):
                delta = self.buffer[DataKeys.REWARDS][:, t, ...] + (
                    self.hparams.gamma * self.buffer[DataKeys.VALUES][:, t + 1, ...]
                    - self.buffer[DataKeys.VALUES][:, t, ...]
                )
                self.buffer[DataKeys.ADVANTAGES][:, t, ...] = prev_advantage = delta + (
                    self.hparams.gamma * self.hparams.gae_lambda * prev_advantage
                )
            self.buffer[DataKeys.RETURNS] = (
                self.buffer[DataKeys.ADVANTAGES] + self.buffer[DataKeys.VALUES]
            )
            std, mean = torch.std_mean(self.buffer[DataKeys.ADVANTAGES])
            self.buffer[DataKeys.ADVANTAGES] = (
                self.buffer[DataKeys.ADVANTAGES] - mean
            ) / (std + 1e-8)

            # Batchify the buffer. Save the last sample for adding it back to the
            # buffer. Remove the last sample afterwards since it contains dummy
            # data.
            final_obs = self.buffer[DataKeys.OBS][:, -1, ...]
            self.buffer = self.buffer[:, :-1, ...]
            views = self.policy.model.apply_view_requirements(self.buffer, kind="all")
            self.buffer = self.buffer.reshape(-1)
            self.buffer[DataKeys.VIEWS] = views

            # Free buffer elements that aren't used for the rest of the step.
            del self.buffer[DataKeys.OBS]
            del self.buffer[DataKeys.REWARDS]
            del self.buffer[DataKeys.VALUES]

            # Main PPO loop.
            step_stats_per_batch: list[StepStats] = []
            loader = DataLoader(
                self.buffer,
                batch_size=self.hparams.sgd_minibatch_size,
                shuffle=self.hparams.shuffle_minibatches,
                collate_fn=lambda x: x,
            )
            for _ in range(self.hparams.num_sgd_iter):
                for minibatch in loader:
                    sample_batch = self.policy.sample(
                        minibatch,
                        kind="all",
                        deterministic=False,
                        inplace=False,
                        requires_grad=True,
                        return_actions=False,
                        return_logp=False,
                        return_values=True,
                        return_views=False,
                    )

                    # Get action distributions and their log probability ratios.
                    curr_action_dist = self.policy.dist_cls(
                        sample_batch[DataKeys.FEATURES], self.policy.model
                    )
                    ratio = torch.exp(
                        curr_action_dist.logp(minibatch[DataKeys.ACTIONS])
                        - minibatch[DataKeys.LOGP]
                    )

                    # Compute main, required losses.
                    vf_loss = torch.mean(
                        torch.clamp(
                            F.smooth_l1_loss(
                                sample_batch[DataKeys.VALUES],
                                minibatch[DataKeys.RETURNS],
                                reduction="none",
                            ),
                            0.0,
                            self.hparams.vf_clip_param,
                        )
                    )
                    surr1 = minibatch[DataKeys.ADVANTAGES] * ratio
                    surr2 = minibatch[DataKeys.ADVANTAGES] * torch.clamp(
                        ratio, 1 - self.hparams.clip_param, 1 + self.hparams.clip_param
                    )
                    if self.hparams.dual_clip_param:
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(
                            clip1,
                            self.hparams.dual_clip_param
                            * minibatch[DataKeys.ADVANTAGES],
                        )
                        policy_loss = torch.where(
                            minibatch[DataKeys.ADVANTAGES] < 0, clip2, clip1
                        ).mean()
                    else:
                        policy_loss = torch.min(
                            surr1,
                            surr2,
                        ).mean()

                    # Maximize entropy, maximize policy actions associated with high advantages,
                    # minimize discounted return estimation error.
                    total_loss = self.hparams.vf_coeff * vf_loss - policy_loss
                    if self.entropy_scheduler.coeff != 0:
                        entropy_loss = curr_action_dist.entropy().mean()
                        total_loss -= self.entropy_scheduler.coeff * entropy_loss
                    else:
                        entropy_loss = torch.tensor([0.0])

                    # Calculate approximate KL divergence for debugging.
                    with torch.no_grad():
                        logp_ratio = (
                            curr_action_dist.logp(minibatch[DataKeys.ACTIONS])
                            - minibatch[DataKeys.LOGP]
                        )
                        kl_div = torch.mean((torch.exp(logp_ratio) - 1) - logp_ratio)

                    # Optimize.
                    self.optimizer.zero_grad()
                    total_loss.backward()  # type: ignore[no-untyped-call]
                    nn.utils.clip_grad_norm_(
                        self.policy.model.parameters(), self.hparams.max_grad_norm
                    )
                    self.optimizer.step()

                    # Update step data.
                    step_stats_per_batch.append(
                        {
                            "coefficients/entropy": self.entropy_scheduler.coeff,
                            "coefficients/vf": self.hparams.vf_coeff,
                            "losses/entropy": float(entropy_loss),
                            "losses/policy": float(policy_loss),
                            "losses/vf": float(vf_loss),
                            "losses/total": float(total_loss),
                            "monitors/kl_div": float(kl_div),
                        }
                    )

            # Update schedulers.
            self.lr_scheduler.step(B * T)
            self.entropy_scheduler.step(B * T)

            # Reset the buffer and buffered flag.
            self.buffer = self.buffer_spec.zero(
                [
                    B,
                    T + 1,
                ]
            )
            self.buffer[DataKeys.OBS][:, -1, ...] = final_obs
            self.state.buffered = False

            # Update algo stats.
            step_stats = pd.DataFrame(step_stats_per_batch).mean(axis=0).to_dict()
        self.state.step_calls += 1
        step_stats["counting/step_calls"] = self.state.step_calls
        step_stats["profiling/step_ms"] = step_timer()
        return step_stats  # type: ignore[no-any-return]

    def to(self, device: Device, /) -> Self:
        """Move the algorithm and its attributes to ``device``."""
        self.buffer_spec = self.buffer_spec.to(device)
        self.buffer = self.buffer.to(device)
        self.env = self.env.to(device)
        self.policy = self.policy.to(device)
        return self
