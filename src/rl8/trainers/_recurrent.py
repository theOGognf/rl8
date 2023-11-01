"""High-level training interfaces."""

from typing import Any

import torch.optim as optim

from ..algorithms import RecurrentAlgorithm
from ..data import Device
from ..distributions import Distribution
from ..env import EnvFactory
from ..models import RecurrentModel, RecurrentModelFactory
from ..schedulers import ScheduleKind
from ._base import GenericTrainerBase


class RecurrentTrainer(GenericTrainerBase[RecurrentAlgorithm]):
    """Higher-level training interface that interops with other tools for
    tracking and saving experiments (i.e., MLflow).

    This is the preferred training interface when training recurrent
    policies in most cases.

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
            if you don't provide one. Defaults to a simple recurrent
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

    """

    def __init__(
        self,
        env_cls: EnvFactory,
        /,
        *,
        env_config: None | dict[str, Any] = None,
        model: None | RecurrentModel = None,
        model_cls: None | RecurrentModelFactory = None,
        model_config: None | dict[str, Any] = None,
        distribution_cls: None | type[Distribution] = None,
        horizon: int = 32,
        horizons_per_env_reset: int = 1,
        num_envs: int = 8192,
        seq_len: int = 4,
        seqs_per_state_reset: int = 8,
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
        super().__init__(
            RecurrentAlgorithm(
                env_cls,
                env_config=env_config,
                model=model,
                model_cls=model_cls,
                model_config=model_config,
                distribution_cls=distribution_cls,
                horizon=horizon,
                horizons_per_env_reset=horizons_per_env_reset,
                num_envs=num_envs,
                seq_len=seq_len,
                seqs_per_state_reset=seqs_per_state_reset,
                optimizer_cls=optimizer_cls,
                optimizer_config=optimizer_config,
                accumulate_grads=accumulate_grads,
                enable_amp=enable_amp,
                lr_schedule=lr_schedule,
                lr_schedule_kind=lr_schedule_kind,
                entropy_coeff=entropy_coeff,
                entropy_coeff_schedule=entropy_coeff_schedule,
                entropy_coeff_schedule_kind=entropy_coeff_schedule_kind,
                gae_lambda=gae_lambda,
                gamma=gamma,
                sgd_minibatch_size=sgd_minibatch_size,
                num_sgd_iters=num_sgd_iters,
                shuffle_minibatches=shuffle_minibatches,
                clip_param=clip_param,
                vf_clip_param=vf_clip_param,
                dual_clip_param=dual_clip_param,
                vf_coeff=vf_coeff,
                max_grad_norm=max_grad_norm,
                device=device,
            )
        )
