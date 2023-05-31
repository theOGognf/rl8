from dataclasses import asdict
from typing import Any

import torch
import torch.amp as amp
import torch.optim as optim
from tensordict import TensorDict
from torch.utils.data import DataLoader

from .._utils import StatTracker, assert_nd_spec, profile_ms
from ..data import (
    CollectStats,
    DataKeys,
    Device,
    RecurrentAlgorithmHparams,
    RecurrentAlgorithmState,
    StepStats,
)
from ..distributions import Distribution
from ..env import Env
from ..models import RecurrentModel
from ..nn import generalized_advantage_estimate, ppo_losses
from ..optimizer import OptimizerWrapper
from ..policies import RecurrentPolicy
from ..schedulers import EntropyScheduler, LRScheduler, ScheduleKind
from ..specs import CompositeSpec, UnboundedContinuousTensorSpec


class RecurrentAlgorithm:
    """An optimized recurrent `PPO`_ algorithm with common tricks for stabilizing
    and accelerating learning.

    Args:
        env_cls: Highly parallelized environment for sampling experiences.
            Instantiated with ``env_config``. Will be stepped for ``horizon``
            each :meth:`RecurrentAlgorithm.collect` call.
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
        distribution_cls: Custom policy action distribution class. An action
            distribution class is provided for you based on the environment
            instance's specs if you don't provide one. Defaults to a categorical
            action distribution for discrete actions and a normal action
            distribution for continuous actions. Complex actions are not
            supported for default action distributions.
        horizon: Number of environment transitions to collect during
            :meth:`RecurrentAlgorithm.collect`. The environment is reset based on
            ``horizons_per_env_reset``. The buffer's size is ``[B, T]`` where ``T`` is
            ``horizon``.
        horizons_per_env_reset: Number of times :meth:`RecurrentAlgorithm.collect` can be
            called before resetting :attr:`RecurrentAlgorithm.env`. Set this to a higher
            number if you want learning to occur across horizons. Leave this
            as the default ``1`` if it doesn't matter that experiences and
            learning only occurs within one horizon.
        num_envs: Number of parallelized simulation environments for the
            environment instance. Passed during the environment's
            instantiation. The buffer's size is ``[B, T]`` where ``B`` is
            ``num_envs``.
        seq_len: Truncated backpropagation through time sequence length.
            Not necessarily the sequence length the recurrent states
            are propagated for prior to being reset. This parameter
            coupled with ``seqs_per_state_reset`` controls how many environment transitions
            are made before recurrent model states are reset or reinitialized.
        seqs_per_state_reset: Number of sequences made within
            :meth:`RecurrentAlgorithm.collect` before recurrent model states
            are reset or reinitialized. Recurrent model states are never reset or
            reinitialized if this parameter is negative.
        optimizer_cls: Custom optimizer class. Defaults to an optimizer
            that doesn't require much tuning.
        optimizer_config: Custom optimizer config unpacked into ``optimizer_cls``
            during optimizer instantiation.
        accumulate_grads: Whether to accumulate gradients using minibatches for each
            epoch prior to stepping the optimizer. Useful for increasing
            the effective batch size while minimizing memory usage.
        enable_amp: Whether to enable Automatic Mixed Precision (AMP) to reduce
            accelerate training and reduce training memory usage.
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
            :attr:`RecurrentAlgorithm.buffer` is split into when updating the policy's model
            in :meth:`RecurrentAlgorithm.step`. It's usually best to maximize the minibatch
            size to reduce the variance associated with updating the policy's model,
            and also to accelerate the computations when learning (assuming a CUDA
            device is being used). If ``None``, the whole buffer is treated as one giant
            batch.
        num_sgd_iters: PPO hyperparameter indicating the number of gradient steps to take
            with the whole :attr:`RecurrentAlgorithm.buffer` when calling :meth:`RecurrentAlgorithm.step`.
        shuffle_minibatches: Whether to shuffle minibatches within :meth:`RecurrentAlgorithm.step`.
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
            within :meth:`RecurrentAlgorithm.step`.
        device: Device :attr:`RecurrentAlgorithm.env`, :attr:`RecurrentAlgorithm.buffer`, and
            :attr:`RecurrentAlgorithm.policy` all reside on.

    Examples:
        Instantiate an algorithm for a dummy environment and update tne underlying
        policy once.

        >>> from rlstack import RecurrentAlgorithm
        >>> from rlstack.env import DiscreteDummyEnv
        >>> algo = RecurrentAlgorithm(DiscreteDummyEnv)
        >>> algo.collect()
        >>> algo.step()  # doctest: +SKIP

    .. _`PPO`: https://arxiv.org/pdf/1707.06347.pdf

    """

    #: Environment experience buffer used for aggregating environment
    #: transition data and policy sample data. The same buffer object
    #: is shared whenever using :meth:`RecurrentAlgorithm.collect` Buffer dimensions
    #: are determined by ``num_envs`` and ``horizon`` args.
    buffer: TensorDict

    #: Tensor spec defining the environment experience buffer components
    #: and dimensions. Used for instantiating :attr:`RecurrentAlgorithm.buffer`
    #: at :class:`RecurrentAlgorithm` instantiation and each :meth:`RecurrentAlgorithm.step`
    #: call.
    buffer_spec: CompositeSpec

    #: Entropy scheduler for updating the ``entropy_coeff`` after each
    #: :meth:`RecurrentAlgorithm.step` call based on the number environment transitions
    #: collected and learned on. By default, the entropy scheduler does not
    #: actually update the entropy coefficient. The entropy scheduler only
    #: updates the entropy coefficient if an ``entropy_coeff_schedule`` is
    #: provided.
    entropy_scheduler: EntropyScheduler

    #: Environment used for experience collection within the
    #: :meth:`RecurrentAlgorithm.collect` method. It's ultimately up to the environment
    #: to make learning efficient by parallelizing simulations.
    env: Env

    #: Recurrent PPO hyperparameters that're constant throughout training
    #: and can drastically affect training performance.
    hparams: RecurrentAlgorithmHparams

    #: Learning rate scheduler for updating `optimizer` learning rate after
    #: each `step` call based on the number of environment transitions
    #: collected and learned on. By default, the learning scheduler does not
    #: actually alter the `optimizer` learning rate (it actually leaves it
    #: constant). The learning rate scheduler only alters the learning rate
    #: if a `learning_rate_schedule` is provided.
    lr_scheduler: LRScheduler

    #: Wrapper around the underlying optimizer for updating the policy's model
    #: that was constructed from ``optimizer_cls`` and ``optimizer_config``.
    #: Handles gradient accumulation and Automatic Mixed Precision (AMP) model
    #: updates. ``optimizer_cls`` defaults to the Adam optimizer.
    optimizer: OptimizerWrapper

    #: Policy constructed from the ``model_cls``, ``model_config``, and
    #: ``distribution_cls`` kwargs. A default policy is constructed according to
    #: the environment's observation and action specs if these policy args
    #: aren't provided. The policy is what does all the action sampling
    #: within :meth:`RecurrentAlgorithm.collect` and is what is updated within
    #: :meth:`RecurrentAlgorithm.step`.
    policy: RecurrentPolicy

    #: Algorithm state for determining when to reset the environment,
    #: when to reset recurrent model states, when the policy can be updated,
    #: and for tracking additional algorithm metrics like method call counts.
    state: RecurrentAlgorithmState

    def __init__(
        self,
        env_cls: type[Env],
        /,
        *,
        env_config: None | dict[str, Any] = None,
        model: None | RecurrentModel = None,
        model_cls: None | type[RecurrentModel] = None,
        model_config: None | dict[str, Any] = None,
        distribution_cls: None | type[Distribution] = None,
        horizon: None | int = 32,
        horizons_per_env_reset: int = 1,
        num_envs: int = 8192,
        seq_len: int = 4,
        seqs_per_state_reset: int = 4,
        optimizer_cls: type[optim.Optimizer] = optim.Adam,
        optimizer_config: None | dict[str, Any] = None,
        accumulate_grads: bool = False,
        enable_amp: bool = False,
        lr_schedule: None | list[tuple[int, float]] = None,
        lr_schedule_kind: ScheduleKind = "step",
        entropy_coeff: float = 0.0,
        entropy_coeff_schedule: None | list[tuple[int, float]] = None,
        entropy_coeff_schedule_kind: ScheduleKind = "step",
        gae_lambda: float = 0.95,
        gamma: float = 0.95,
        sgd_minibatch_size: None | int = None,
        num_sgd_iters: int = 4,
        shuffle_minibatches: bool = True,
        clip_param: float = 0.2,
        vf_clip_param: float = 5.0,
        dual_clip_param: None | float = None,
        vf_coeff: float = 1.0,
        max_grad_norm: float = 5.0,
        device: Device = "cpu",
    ) -> None:
        self.env = env_cls(num_envs, config=env_config, device=device)
        assert_nd_spec(self.env.observation_spec)
        assert_nd_spec(self.env.action_spec)
        self.policy = RecurrentPolicy(
            self.env.observation_spec,
            self.env.action_spec,
            model=model,
            model_cls=model_cls,
            model_config=model_config,
            distribution_cls=distribution_cls,
            device=device,
        )
        max_horizon = self.env.max_horizon if hasattr(self.env, "max_horizon") else 32
        horizon = min(horizon, max_horizon) if horizon else max_horizon
        self.buffer_spec = CompositeSpec(  # type: ignore[no-untyped-call]
            {
                DataKeys.OBS: self.env.observation_spec,
                DataKeys.STATES: self.policy.state_spec,
                DataKeys.REWARDS: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.ACTIONS: self.env.action_spec,
                DataKeys.LOGP: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.VALUES: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.ADVANTAGES: UnboundedContinuousTensorSpec(1, device=device),
                DataKeys.RETURNS: UnboundedContinuousTensorSpec(1, device=device),
            },
        ).to(device)
        self.buffer = self.buffer_spec.zero([num_envs, horizon + 1])
        optimizer_config = optimizer_config or {"lr": 1e-3}
        optimizer = optimizer_cls(self.policy.model.parameters(), **optimizer_config)
        self.lr_scheduler = LRScheduler(
            optimizer, schedule=lr_schedule, kind=lr_schedule_kind
        )
        self.entropy_scheduler = EntropyScheduler(
            entropy_coeff,
            schedule=entropy_coeff_schedule,
            kind=entropy_coeff_schedule_kind,
        )
        sgd_minibatch_size = (
            sgd_minibatch_size
            if sgd_minibatch_size
            else num_envs * (horizon // seq_len)
        )
        self.hparams = RecurrentAlgorithmHparams(
            accumulate_grads=accumulate_grads,
            clip_param=clip_param,
            device=str(device),
            dual_clip_param=dual_clip_param,
            enable_amp=enable_amp,
            gae_lambda=gae_lambda,
            gamma=gamma,
            horizon=horizon,
            horizons_per_env_reset=horizons_per_env_reset,
            max_grad_norm=max_grad_norm,
            num_envs=num_envs,
            num_sgd_iters=num_sgd_iters,
            seq_len=seq_len,
            seqs_per_state_reset=seqs_per_state_reset,
            sgd_minibatch_size=sgd_minibatch_size,
            shuffle_minibatches=shuffle_minibatches,
            vf_clip_param=vf_clip_param,
            vf_coeff=vf_coeff,
        ).validate()
        self.state = RecurrentAlgorithmState()
        self.optimizer = OptimizerWrapper(
            optimizer,
            enable_amp=enable_amp,
            grad_accumulation_steps=self.hparams.num_minibatches
            if accumulate_grads
            else 1,
        )

    def collect(
        self, *, env_config: None | dict[str, Any] = None, deterministic: bool = False
    ) -> CollectStats:
        """Collect environment transitions and policy samples in a buffer.

        This is one of the main :class:`RecurrentAlgorithm` methods. This is usually
        called immediately prior to :meth:`RecurrentAlgorithm.step` to collect
        experiences used for learning.

        The environment is reset immediately prior to collecting
        transitions according to ``horizons_per_env_reset``. If
        the environment isn't reset, then the last observation is used as
        the initial observation.

        This method sets the ``buffered`` flag to enable calling
        of :meth:`RecurrentAlgorithm.step` so it isn't called with dummy data.

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
            # Gather initial observation and states.
            if not (self.state.horizons % self.hparams.horizons_per_env_reset):
                self.buffer[DataKeys.OBS][:, 0, ...] = self.env.reset(config=env_config)
            else:
                self.buffer[DataKeys.OBS][:, 0, ...] = self.buffer[DataKeys.OBS][
                    :, -1, ...
                ]
            self.buffer[DataKeys.STATES][:, 0, ...] = self.buffer[DataKeys.STATES][
                :, -1, ...
            ]

            for t in range(self.hparams.horizon):
                if self.state.seqs and self.hparams.seqs_per_state_reset <= 0:
                    pass
                elif not (t % self.hparams.seq_len) and not (
                    self.state.seqs % self.hparams.seqs_per_state_reset
                ):
                    self.buffer[DataKeys.STATES][:, t, ...] = self.policy.init_states(
                        self.hparams.num_envs
                    )

                # Sample the policy and step the environment.
                in_batch = self.buffer[:, t : (t + 1), ...]
                in_states = self.buffer[DataKeys.STATES][:, t : (t + 1), ...]
                sample_batch, sample_states = self.policy.sample(
                    in_batch,
                    in_states,
                    deterministic=deterministic,
                    inplace=False,
                    keepdim=False,
                    requires_grad=False,
                    return_actions=True,
                    return_logp=True,
                    return_values=True,
                )
                out_batch = self.env.step(sample_batch[DataKeys.ACTIONS])

                # Update the buffer using sampled policy data and environment
                # transition data.
                self.buffer[DataKeys.ACTIONS][:, t, ...] = sample_batch[
                    DataKeys.ACTIONS
                ]
                self.buffer[DataKeys.LOGP][:, t, ...] = sample_batch[DataKeys.LOGP]
                self.buffer[DataKeys.VALUES][:, t, ...] = sample_batch[DataKeys.VALUES]
                self.buffer[DataKeys.REWARDS][:, t, ...] = out_batch[DataKeys.REWARDS]
                self.buffer[DataKeys.OBS][:, t + 1, ...] = out_batch[DataKeys.OBS]
                self.buffer[DataKeys.STATES][:, t + 1, ...] = sample_states

                if t and not (t % self.hparams.seq_len):
                    self.state.seqs += 1

            # Sample features and value function at last observation.
            in_batch = self.buffer[:, -1:, ...]
            in_states = self.buffer[DataKeys.STATES][:, -1:, ...]
            sample_batch, _ = self.policy.sample(
                in_batch,
                in_states,
                deterministic=deterministic,
                inplace=False,
                keepdim=False,
                requires_grad=False,
                return_actions=False,
                return_logp=False,
                return_values=True,
            )
            self.buffer[DataKeys.VALUES][:, -1, ...] = sample_batch[DataKeys.VALUES]

            self.state.horizons += 1
            self.state.buffered = True

            # Aggregate some metrics.
            rewards = self.buffer[DataKeys.REWARDS][:, 1:-1, ...]
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
        self.state.total_steps += self.hparams.num_envs * self.hparams.horizon
        collect_stats["counting/collect_calls"] = self.state.collect_calls
        collect_stats["counting/horizons"] = self.state.horizons
        collect_stats["counting/total_steps"] = self.state.total_steps
        collect_stats["profiling/collect_ms"] = collect_timer()
        return collect_stats

    @property
    def params(self) -> dict[str, Any]:
        """Return algorithm parameters."""
        return {
            "env_cls": self.env.__class__.__name__,
            "model_cls": self.policy.model.__class__.__name__,
            "distribution_cls": self.policy.distribution_cls.__name__,
            "optimizer_cls": self.optimizer.optimizer.__class__.__name__,
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
            # Generalized Advantage Estimation (GAE) and returns bootstrapping.
            self.buffer = generalized_advantage_estimate(
                self.buffer,
                gae_lambda=self.hparams.gae_lambda,
                gamma=self.hparams.gamma,
                inplace=True,
                normalize=True,
                return_returns=True,
            )

            # Batchify the buffer. Save the last sample for adding it back to the
            # buffer. Remove the last sample afterwards since it contains dummy
            # data.
            final_obs = self.buffer[DataKeys.OBS][:, -1, ...]
            final_states = self.buffer[DataKeys.STATES][:, -1, ...]
            self.buffer = self.buffer[:, :-1, ...]
            self.buffer = self.buffer.reshape(-1, self.hparams.seq_len)

            # Free buffer elements that aren't used for the rest of the step.
            del self.buffer[DataKeys.REWARDS]
            del self.buffer[DataKeys.VALUES]

            # Main PPO loop.
            device_type = "cuda" if str(self.hparams.device) != "cpu" else "cpu"
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
            loader = DataLoader(
                self.buffer,
                batch_size=self.hparams.sgd_minibatch_size,
                shuffle=self.hparams.shuffle_minibatches,
                collate_fn=lambda x: x,
            )
            for _ in range(self.hparams.num_sgd_iters):
                for minibatch in loader:
                    with amp.autocast(
                        device_type,
                        enabled=self.hparams.enable_amp,
                    ):
                        sample_batch, _ = self.policy.sample(
                            minibatch,
                            minibatch[DataKeys.STATES],
                            deterministic=False,
                            inplace=False,
                            keepdim=False,
                            requires_grad=True,
                            return_actions=False,
                            return_logp=False,
                            return_values=True,
                        )
                        minibatch = minibatch.reshape(-1)

                        # Get action distributions and their log probability ratios.
                        curr_action_dist = self.policy.distribution_cls(
                            sample_batch[DataKeys.FEATURES], self.policy.model
                        )
                        losses = ppo_losses(
                            minibatch,
                            sample_batch,
                            curr_action_dist,
                            clip_param=self.hparams.clip_param,
                            dual_clip_param=self.hparams.dual_clip_param,
                            entropy_coeff=self.entropy_scheduler.coeff,
                            vf_clip_param=self.hparams.vf_clip_param,
                            vf_coeff=self.hparams.vf_coeff,
                        )
                        losses = losses.apply(lambda x: x / grad_accumulation_steps)

                    # Calculate approximate KL divergence for debugging.
                    with torch.no_grad():
                        logp_ratio = (
                            curr_action_dist.logp(minibatch[DataKeys.ACTIONS])
                            - minibatch[DataKeys.LOGP]
                        )
                        kl_div = (
                            torch.mean((torch.exp(logp_ratio) - 1) - logp_ratio)
                            / grad_accumulation_steps
                        )

                    # Optimize.
                    stepped = self.optimizer.step(
                        losses["total"],
                        self.policy.model.parameters(),
                        max_grad_norm=self.hparams.max_grad_norm,
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
                            "monitors/kl_div": float(kl_div),
                        },
                        reduce=stepped,
                    )

            # Update schedulers.
            self.lr_scheduler.step(self.hparams.num_envs * self.hparams.horizon)
            self.entropy_scheduler.step(self.hparams.num_envs * self.hparams.horizon)

            # Reset the buffer and buffered flag.
            self.buffer = self.buffer_spec.zero(
                [
                    self.hparams.num_envs,
                    self.hparams.horizon + 1,
                ]
            )
            self.buffer[DataKeys.OBS][:, -1, ...] = final_obs
            self.buffer[DataKeys.STATES][:, -1, ...] = final_states
            self.state.buffered = False

            # Update algo stats.
            step_stats = stat_tracker.items()
        self.state.step_calls += 1
        step_stats["counting/step_calls"] = self.state.step_calls
        step_stats["profiling/step_ms"] = step_timer()
        return step_stats  # type: ignore[return-value]
