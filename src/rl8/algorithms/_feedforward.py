from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

from .._utils import Batcher, StatTracker, assert_nd_spec, profile_ms
from ..data import (
    AlgorithmHparams,
    AlgorithmState,
    CollectStats,
    DataKeys,
    Device,
    StepStats,
)
from ..distributions import Distribution
from ..env import EnvFactory
from ..models import Model, ModelFactory
from ..nn import generalized_advantage_estimate, ppo_losses
from ..policies import Policy
from ..schedulers import EntropyScheduler, LRScheduler, ScheduleKind
from ._base import GenericAlgorithmBase


@dataclass
class AlgorithmConfig:
    """Algorith config for building a feedforward PPO algorithm."""

    #: Model instance to use. Mutually exclusive with ``model_cls``.
    model: None | Model = None

    #: Optional custom policy model definition. A model class
    #: is provided for you based on the environment instance's specs
    #: if you don't provide one. Defaults to a simple feedforward
    #: neural network.
    model_cls: None | ModelFactory = None

    #: Optional policy model config unpacked into the model
    #: during instantiation.
    model_config: None | dict[str, Any] = None

    #: Custom policy action distribution class.
    #: If not provided, an action distribution class is inferred from the
    #: environment specs. Defaults to a categorical distribution for discrete
    #: actions and a normal distribution for continuous actions. Complex
    #: actions are not supported by default distributions.
    distribution_cls: None | type[Distribution] = None

    #: Number of environment transitions to collect during :meth:`Algorithm.collect`.
    #: The environment resets according to ``horizons_per_env_reset``.
    #: Buffer size is [B, T] where T = horizon.
    horizon: int = 32

    #: Number of times :meth:`Algorithm.collect` can be called before resetting
    #: :attr:`Algorithm.env`. Increase this for cross-horizon learning. Default
    #: 1 resets after every horizon.
    horizons_per_env_reset: int = 1

    #: Number of parallelized environment instances.
    #: Determines buffer size [B, T] where B = num_envs.
    num_envs: int = 8192

    #: Custom optimizer class. Defaults to a simple, low-tuning optimizer.
    optimizer_cls: type[optim.Optimizer] = optim.Adam

    #: Configuration passed to the optimizer during instantiation.
    optimizer_config: None | dict[str, Any] = None

    #: Whether to accumulate gradients across minibatches before stepping the
    #: optimizer. Increases effective batch size while minimizing memory usage.
    accumulate_grads: bool = False

    #: Whether to enable Automatic Mixed Precision (AMP) for faster and more
    #: memory-efficient training.
    enable_amp: bool = False

    #: Optional schedule controlling the optimizer's learning rate over
    #: environment transitions. Keeps learning rate constant if not provided.
    lr_schedule: None | list[tuple[int, float]] = None

    #: Learning rate scheduler type if lr_schedule is provided.
    #: Options: ``"step"`` (jump and hold) or ``"interp"`` (interpolate between
    #: values).
    lr_schedule_kind: ScheduleKind = "step"

    #: Entropy coefficient weight in total loss.
    #: Ignored if ``entropy_coeff_schedule`` is provided.
    entropy_coeff: float = 0.0

    #: Optional schedule overriding entropy_coeff based on number of environment
    #: transitions.
    entropy_coeff_schedule: None | list[tuple[int, float]] = None

    #: Entropy scheduler type. Options:
    #: ``"step"``: jump and hold, ``"interp"``: interpolate between values.
    entropy_coeff_schedule_kind: ScheduleKind = "step"

    #: Generalized Advantage Estimation (GAE) λ parameter for controlling the
    #: variance and bias tradeoff when estimating the state value function
    #: from collected environment transitions. A higher value allows higher
    #: variance while a lower value allows higher bias estimation but lower
    #: variance.
    gae_lambda: float = 0.95

    #: Discount reward factor often used in the Bellman operator for controlling
    #: the variance and bias tradeoff in collected experienced rewards. Note,
    #: this does not control the bias/variance of the state value estimation
    #: and only controls the weight future rewards have on the total discounted
    #: return.
    gamma: float = 0.95

    #: PPO hyperparameter for minibatch size during policy update.
    #: Larger minibatches reduce update variance and accelerate CUDA computations.
    #: If ``None``, the entire buffer is treated as one batch.
    sgd_minibatch_size: None | int = None

    #: PPO hyperparameter for number of SGD iterations over the collected buffer.
    num_sgd_iters: int = 4

    #: Whether to shuffle minibatches within :meth:`Algorithm.step`. Recommended, but
    #: not necessary if the minibatch size is large enough (e.g., the buffer is the
    #: batch).
    shuffle_minibatches: bool = True

    #: PPO hyperparameter indicating the max distance the policy can update away from
    #: previously collected policy sample data with respect to likelihoods of taking
    #: actions conditioned on observations. This is the main innovation of PPO.
    clip_param: float = 0.2

    #: PPO hyperparameter similar to ``clip_param`` but for the value function estimate.
    #: A measure of max distance the model's value function is allowed to update away
    #: from previous value function samples.
    vf_clip_param: float = 5.0

    #: PPO hyperparameter that clips like ``clip_param`` but when advantage estimations
    #: are negative. Helps prevent instability for continuous action spaces when
    #: policies are making large updates. Leave ``None`` for this clip to not apply.
    #: Otherwise, typical values are around ``5``.
    dual_clip_param: None | float = None

    #: Value function loss component weight. Only needs to be tuned when the policy
    #: and value function share parameters.
    vf_coeff: float = 1.0

    #: Target maximum KL divergence when updating the policy. If approximate KL
    #: divergence is greater than this value, then policy updates stop early for
    #: that algorithm step. If this is left `None then early stopping doesn't occur.
    #: A higher value means the policy is allowed to diverge more from the previous
    #: policy during updates.
    target_kl_div: None | float = None

    #: Max gradient norm allowed when updating the policy's model within
    #: :meth:`Algorithm.step`.
    max_grad_norm: float = 5.0

    #: Whether to normalize advantages computed for GAE using the batch's
    #: mean and standard deviation. This has been shown to generally improve
    #: convergence speed and performance and should usually be ``True``.
    normalize_advantages: bool = True

    #: Whether to normalize rewards using reversed discounted returns as
    #: from https://arxiv.org/pdf/2005.12729.pdf. Reward normalization,
    #: although not exactly correct and optimal, typically improves
    #: convergence speed and performance and should usually be ``True``.
    normalize_rewards: bool = True

    #: Device :attr:`Algorithm.env`, :attr:`Algorithm.buffer`, and
    #: :attr:`Algorithm.policy` all reside on.
    device: Device | Literal["auto"] = "auto"

    def build(self, env_cls: EnvFactory) -> "Algorithm":
        """Build and validate an :class:Algorithm` from a config."""
        algo = Algorithm(env_cls, config=self)
        algo.validate()
        return algo


class Algorithm(GenericAlgorithmBase[AlgorithmHparams, AlgorithmState, Policy]):
    """An optimized feedforward `PPO`_ algorithm with common tricks for
    stabilizing and accelerating learning.

    Args:
        env_cls: Highly parallelized environment for sampling experiences.
            Will be stepped for ``horizon`` each :meth:`Algorithm.collect` call.
        config: Algorithm config for building a feedforward PPO
            algorithm. See :class:`AlgorithmConfig` for all parameters.

    Examples:
        Instantiate an algorithm for a dummy environment and update the underlying
        policy once.

        >>> from rl8 import AlgorithmConfig
        >>> from rl8.env import DiscreteDummyEnv
        >>> algo = AlgorithmConfig().build(DiscreteDummyEnv)
        >>> algo.collect()  # doctest: +SKIP
        >>> algo.step()  # doctest: +SKIP

    .. _`PPO`: https://arxiv.org/pdf/1707.06347.pdf

    """

    def __init__(
        self, env_cls: EnvFactory, /, config: None | AlgorithmConfig = None
    ) -> None:
        config = config or AlgorithmConfig()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            if config.device == "auto"
            else config.device
        )
        max_num_envs = (
            env_cls.max_num_envs
            if hasattr(env_cls, "max_num_envs")
            else config.num_envs
        )
        num_envs = min(config.num_envs, max_num_envs)
        max_horizon = (
            env_cls.max_horizon if hasattr(env_cls, "max_horizon") else 1_000_000
        )
        horizon = min(config.horizon, max_horizon)
        self.env = env_cls(num_envs, horizon, device=device)
        assert_nd_spec(self.env.observation_spec)
        assert_nd_spec(self.env.action_spec)
        self.policy = Policy(
            self.env.observation_spec,
            self.env.action_spec,
            model=config.model,
            model_cls=config.model_cls,
            model_config=config.model_config,
            distribution_cls=config.distribution_cls,
            device=device,
        )
        self.buffer_spec = CompositeSpec(
            {
                DataKeys.OBS: self.env.observation_spec,
                DataKeys.REWARDS: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.ACTIONS: self.env.action_spec,
                DataKeys.LOGP: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.VALUES: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.ADVANTAGES: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.RETURNS: UnboundedContinuousTensorSpec(1, device=device),
            },
        )
        if config.normalize_rewards:
            self.buffer_spec.set(
                DataKeys.REVERSED_DISCOUNTED_RETURNS,
                UnboundedContinuousTensorSpec(1, device=device),
            )
        self.buffer_spec = self.buffer_spec.to(device)
        self.buffer = self.buffer_spec.zero([num_envs, horizon + 1])
        optimizer_config = config.optimizer_config or {"lr": 1e-3}
        optimizer = config.optimizer_cls(
            self.policy.model.parameters(), **optimizer_config
        )
        self.lr_scheduler = LRScheduler(
            optimizer,
            schedule=config.lr_schedule,
            kind=config.lr_schedule_kind,
        )
        self.entropy_scheduler = EntropyScheduler(
            config.entropy_coeff,
            schedule=config.entropy_coeff_schedule,
            kind=config.entropy_coeff_schedule_kind,
        )
        sgd_minibatch_size = (
            config.sgd_minibatch_size
            if config.sgd_minibatch_size
            else num_envs * horizon
        )
        self.hparams = AlgorithmHparams(
            accumulate_grads=config.accumulate_grads,
            clip_param=config.clip_param,
            device=str(device),
            dual_clip_param=config.dual_clip_param,
            enable_amp=config.enable_amp,
            gae_lambda=config.gae_lambda,
            gamma=config.gamma,
            horizon=horizon,
            horizons_per_env_reset=config.horizons_per_env_reset,
            max_grad_norm=config.max_grad_norm,
            normalize_advantages=config.normalize_advantages,
            normalize_rewards=config.normalize_rewards,
            num_envs=num_envs,
            num_sgd_iters=config.num_sgd_iters,
            sgd_minibatch_size=sgd_minibatch_size,
            shuffle_minibatches=config.shuffle_minibatches,
            target_kl_div=config.target_kl_div,
            vf_clip_param=config.vf_clip_param,
            vf_coeff=config.vf_coeff,
        ).validate()
        self.state = AlgorithmState()
        self.optimizer = optimizer
        self.grad_scaler = GradScaler(enabled=config.enable_amp)

    def collect(
        self,
        *,
        env_config: None | dict[str, Any] = None,
        deterministic: bool = False,
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
            # Gather initial observation.
            env_was_reset = False
            if self.state.horizons and self.hparams.horizons_per_env_reset < 0:
                self.buffer[DataKeys.OBS][:, 0, ...] = self.buffer[DataKeys.OBS][
                    :, -1, ...
                ]
                if self.hparams.normalize_rewards:
                    self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][
                        :, 0, ...
                    ] = self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][:, -1, ...]
            elif not (self.state.horizons % self.hparams.horizons_per_env_reset):
                self.buffer[DataKeys.OBS][:, 0, ...] = self.env.reset(config=env_config)
                env_was_reset = True
                if self.hparams.normalize_rewards:
                    self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][:, 0, ...] = 0.0
            else:
                self.buffer[DataKeys.OBS][:, 0, ...] = self.buffer[DataKeys.OBS][
                    :, -1, ...
                ]
                if self.hparams.normalize_rewards:
                    self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][
                        :, 0, ...
                    ] = self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][:, -1, ...]

            for t in range(self.hparams.horizon):
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

                # Getting reversed discounted returns for normalizing reward
                # scale during GAE. This isn't exactly correct according to
                # theory but works in practice.
                if self.hparams.normalize_rewards:
                    self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][:, t + 1, ...] = (
                        self.hparams.gamma
                        * self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][:, t, ...]
                        + out_batch[DataKeys.REWARDS]
                    )

                # Update the buffer using sampled policy data and environment
                # transition data.
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
            self.buffer[DataKeys.VALUES][:, -1, ...] = sample_batch[DataKeys.VALUES]

            # Aggregate some metrics.
            rewards = self.buffer[DataKeys.REWARDS][:, :-1, ...]
            returns = torch.sum(rewards, dim=1)
            returns_std, returns_mean = torch.std_mean(returns)
            rewards_std, rewards_mean = torch.std_mean(rewards)
            collect_stats: CollectStats = {
                "returns/min": float(torch.min(returns)),
                "returns/max": float(torch.max(returns)),
                "returns/mean": float(returns_mean),
                "returns/std": float(returns_std),
                "rewards/min": float(torch.min(rewards)),
                "rewards/max": float(torch.max(rewards)),
                "rewards/mean": float(rewards_mean),
                "rewards/std": float(rewards_std),
            }

            self.state.horizons += 1
            self.state.buffered = True
            self.state.reward_scale = (
                float(
                    torch.std(
                        self.buffer[DataKeys.REVERSED_DISCOUNTED_RETURNS][:, 1:, ...]
                    )
                )
                if self.hparams.normalize_rewards
                else 1.0
            )

        collect_stats["env/resets"] = self.hparams.num_envs * int(env_was_reset)
        collect_stats["env/steps"] = self.hparams.num_envs * self.hparams.horizon
        collect_stats["profiling/collect_ms"] = collect_timer()
        return collect_stats

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
            # Generalized Advantage Estimation (GAE) and returns bootstrapping.
            self.buffer = generalized_advantage_estimate(
                self.buffer,
                gae_lambda=self.hparams.gae_lambda,
                gamma=self.hparams.gamma,
                inplace=True,
                normalize_advantages=self.hparams.normalize_advantages,
                return_returns=True,
                reward_scale=self.state.reward_scale,
            )

            # Batchify the buffer. Save the last sample for adding it back to the
            # buffer. Remove the last sample afterwards since it contains dummy
            # data.
            final_obs = self.buffer[DataKeys.OBS][:, -1, ...]
            self.buffer = self.buffer[:, :-1, ...]
            views = self.policy.model.apply_view_requirements(self.buffer, kind="all")

            # Free buffer elements that aren't used for the rest of the step.
            del self.buffer[DataKeys.OBS]
            del self.buffer[DataKeys.REWARDS]
            del self.buffer[DataKeys.VALUES]

            self.buffer = self.buffer.reshape(-1)
            self.buffer[DataKeys.VIEWS] = views

            # Main PPO loop.
            grad_accumulation_steps = (
                self.hparams.num_minibatches if self.hparams.accumulate_grads else 1
            )
            stat_tracker = StatTracker(
                [
                    "coefficients/entropy",
                    "coefficients/vf",
                    "losses/entropy",
                    "losses/policy",
                    "losses/vf",
                    "losses/total",
                    "monitors/kl_div",
                ],
                sum_keys=[
                    "losses/entropy",
                    "losses/policy",
                    "losses/vf",
                    "losses/total",
                    "monitors/kl_div",
                ],
            )
            batcher = Batcher(
                self.buffer,
                batch_size=self.hparams.sgd_minibatch_size,
                shuffle=self.hparams.shuffle_minibatches,
            )
            stop_early = False
            for _ in range(self.hparams.num_sgd_iters):
                for i, buffer_batch in enumerate(batcher):
                    step_this_batch = (i + 1) % grad_accumulation_steps == 0
                    with amp.autocast(
                        self.hparams.device_type,
                        enabled=self.hparams.enable_amp,
                    ):
                        sample_batch = self.policy.sample(
                            buffer_batch,
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
                        curr_action_dist = self.policy.distribution_cls(
                            sample_batch[DataKeys.FEATURES], self.policy.model
                        )
                        losses = ppo_losses(
                            buffer_batch,
                            sample_batch,
                            curr_action_dist,
                            clip_param=self.hparams.clip_param,
                            dual_clip_param=self.hparams.dual_clip_param,
                            entropy_coeff=self.entropy_scheduler.coeff,
                            vf_clip_param=self.hparams.vf_clip_param,
                            vf_coeff=self.hparams.vf_coeff,
                        )
                        losses = losses.apply(lambda x: x / grad_accumulation_steps)

                    # Calculate approximate KL divergence for early-stopping and
                    # debugging. Early-stopping is per-batch and can't be done with
                    # gradient accumulation (hence approximate KL isn't compared to
                    # target KL with the number of gradient accumulation steps
                    # factor).
                    with torch.no_grad():
                        logp_ratio = (
                            curr_action_dist.logp(buffer_batch[DataKeys.ACTIONS])
                            - buffer_batch[DataKeys.LOGP]
                        )
                        approximate_kl_div = float(
                            torch.mean((torch.exp(logp_ratio) - 1) - logp_ratio)
                        )

                    # Update step data.
                    stat_tracker.update(
                        {
                            "coefficients/entropy": self.entropy_scheduler.coeff,
                            "coefficients/vf": self.hparams.vf_coeff,
                            "losses/entropy": float(losses["entropy"]),
                            "losses/policy": float(losses["policy"]),
                            "losses/vf": float(losses["vf"]),
                            "losses/total": float(losses["total"]),
                            "monitors/kl_div": approximate_kl_div
                            / grad_accumulation_steps,
                        },
                        reduce=step_this_batch,
                    )

                    # Early stopping using approximate KL divergence.
                    if (
                        self.hparams.target_kl_div is not None
                        and approximate_kl_div > 1.5 * self.hparams.target_kl_div
                    ):
                        stop_early = True
                        break

                    # Optimize.
                    self.grad_scaler.scale(losses["total"]).backward()
                    if step_this_batch:
                        self.grad_scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.policy.model.parameters(), self.hparams.max_grad_norm
                        )
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.optimizer.zero_grad()

                if stop_early:
                    break

            # Update schedulers.
            self.lr_scheduler.step(self.hparams.num_envs * self.state.horizons)
            self.entropy_scheduler.step(self.hparams.num_envs * self.state.horizons)

            # Reset the buffer and buffered flag.
            self.buffer = self.buffer_spec.zero(
                [
                    self.hparams.num_envs,
                    self.hparams.horizon + 1,
                ]
            )
            self.buffer[DataKeys.OBS][:, -1, ...] = final_obs
            self.state.buffered = False

            # Update algo stats.
            step_stats = stat_tracker.items()
        step_stats["profiling/step_ms"] = step_timer()
        return step_stats  # type: ignore[return-value]

    def validate(self) -> None:
        """Do some validation on all the tensor/tensordict shapes within
        the algorithm.

        Helpful when the algorithm is throwing an error on mismatched tensor/tensordict
        sizes. Call this at least once before running the algorithm for peace of
        mind.

        """
        # Check initial observation.
        obs = self.env.reset()

        self.env.observation_spec.assert_is_in(obs)
        try:
            self.buffer[DataKeys.OBS][:, 0, ...] = obs
        except RuntimeError as e:
            raise AssertionError(
                f"The observation from {self.env.reset.__qualname__} doesn't match the"
                " observation spec shape."
            ) from e

        # Sample the policy and check all outputs.
        in_batch = self.buffer[:, :1, ...]
        sample_batch = self.policy.sample(
            in_batch,
            kind="last",
            deterministic=False,
            inplace=False,
            requires_grad=False,
            return_actions=True,
            return_logp=True,
            return_values=True,
        )

        actions = sample_batch[DataKeys.ACTIONS]
        assert actions.ndim >= 2, (
            "Actions must be at least 2D and have shape ``[N, ...]`` (where ``N`` is"
            " the number of independent elements or environment instances, and ``...``"
            " is any number of additional dimensions)."
        )
        self.env.action_spec.assert_is_in(actions)
        try:
            self.buffer[DataKeys.ACTIONS][:, 0, ...] = sample_batch[DataKeys.ACTIONS]
        except RuntimeError as e:
            raise AssertionError(
                "The action sampled from the policy doesn't match the action spec."
            ) from e

        assert sample_batch[DataKeys.LOGP].shape == torch.Size(
            [self.hparams.num_envs, 1]
        ), (
            "Action log probabilities must be 2D and have shape ``[N, 1]`` (where ``N``"
            " is the number of independent elements or environment instances)."
        )

        assert sample_batch[DataKeys.VALUES].shape == torch.Size(
            [self.hparams.num_envs, 1]
        ), (
            "Expected value estimates must be 2D and have shape ``[N, 1]`` (where ``N``"
            " is the number of independent elements or environment instances)."
        )

        # Step the environment and check everything once more.
        out_batch = self.env.step(actions)

        obs = out_batch[DataKeys.OBS]
        self.env.observation_spec.assert_is_in(obs)
        try:
            self.buffer[DataKeys.OBS][:, 1, ...] = obs
        except RuntimeError as e:
            raise AssertionError(
                f"The observation from {self.env.step.__qualname__} doesn't match the"
                " observation spec shape."
            ) from e

        assert out_batch[DataKeys.REWARDS].shape == torch.Size(
            [self.hparams.num_envs, 1]
        ), (
            "Rewards must be 2D and have shape ``[N, 1]`` (where ``N`` is the number of"
            " independent elements or environment instances)."
        )
