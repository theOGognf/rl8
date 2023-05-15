"""Definitions related to data passed between algorithm modules."""

from dataclasses import dataclass
from typing import Literal, TypedDict

import torch

Device = str | torch.device


class DataKeys:
    """Collection of common identifiers for elements within batches of data."""

    #: Key denoting observations from the environment.
    #: Typically processed by a policy model.
    OBS = "obs"

    #: Key denoting rewards from the environment.
    #: Typically used by a learning algorithm.
    REWARDS = "rewards"

    #: Key denoting discounted returns.
    RETURNS = "returns"

    #: Key denoting features output from a policy model.
    #: Typically processed by a policy action distribution.
    FEATURES = "features"

    #: Key denoting features output by a policy action distribution.
    #: Usually propagated through an environment.
    ACTIONS = "actions"

    #: Key denoting the log probability of taking `actions` with feature
    #: and a model. Typically used by learning algorithms.
    LOGP = "logp"

    #: Key denoting value function approximation from a policy model.
    #: Typically used by learning algorithms or for analyzing a trained model.
    VALUES = "values"

    #: Key denoting elements that're inputs to a model and have corresponding
    #: "padding_mask" elements.
    INPUTS = "inputs"

    #: Key denoting elements that're used for indicating padded elements
    #: with respect to elements corresponding to an "inputs" key.
    PADDING_MASK = "padding_mask"

    #: Key denoting view requirements applied to another key. These are
    #: the preprocessed inputs to a model.
    VIEWS = "views"

    #: Key denoting advantages (action value function baselined by the state
    #: value function).
    ADVANTAGES = "advantages"

    #: Key denoting recurrent model states. This key is usually coupled with
    #: the ``"hidden_states"`` and/or ``"cell_states"`` keys. This key is
    #: used as recurrent model inputs and recurrent model outputs.
    STATES = "states"

    #: Key denoting recurrent model states. This key is usually coupled with
    #: ``"cell_states"`` and is usually nested under ``"states"``. This key is
    #: used as recurrent model inputs and recurrent model outputs.
    HIDDEN_STATES = "hidden_states"

    #: Key denoting recurrent model states. This key is usually coupled with
    #: ``"hidden_states"`` and is usually nested under ``"states"``. This key is
    #: used as recurrent model inputs and recurrent model outputs.
    CELL_STATES = "cell_states"


@dataclass(frozen=True, kw_only=True)
class AlgorithmHparams:
    """Feedforward PPO hyperparameters that're held constant throughout
    training and can drastically impact training performance.

    Also does some basic hyperparameter validation for convenience.

    """

    #: PPO hyperparameter indicating the max distance the policy can
    #: update away from previously collected policy sample data with
    #: respect to likelihoods of taking actions conditioned on
    #: observations. This is the main innovation of PPO.
    clip_param: float

    #: PPO hyperparameter that clips like :attr:`Algorithm.clip_param` but when
    #: advantage estimations are negative. Helps prevent instability for
    #: continuous action spaces when policies are making large updates.
    dual_clip_param: None | float

    #: Generalized Advantage Estimation (GAE) hyperparameter for controlling
    #: the variance and bias tradeoff when estimating the state value
    #: function from collected environment transitions. A higher value
    #: allows higher variance while a lower value allows higher bias
    #: estimation but lower variance.
    gae_lambda: float

    #: Discount reward factor often used in the Bellman operator for
    #: controlling the variance and bias tradeoff in collected experienced
    #: rewards. Note, this does not control the bias/variance of the
    #: state value estimation and only controls the weight future rewards
    #: have on the total discounted return.
    gamma: float

    #: Number of transitions for each environment for each
    #: :meth:`Algorithm.collect` call prior to calling
    #: :meth:`Algorithm.step`. This hyperparemeter coupled with
    #: :attr:`AlgorithmHparams.horizons_per_env_reset` controls
    #: how many environment transitions are made per environment
    #: before each environment is reset.
    horizon: int

    #: Number of times :meth:`Algorithm.collect` can be called before
    #: resetting :attr:`Algorithm.env`. Set this to a higher number if you
    #: want learning to occur across horizons. Leave this as the default
    #: ``1`` if it doesn't matter that experiences and learning only occurs
    #: within one horizon.
    horizons_per_env_reset: int

    #: Max gradient norm allowed when updating the policy's model within
    #: :meth:`Algorithm.step`.
    max_grad_norm: float

    #: PPO hyperparameter indicating the number of gradient steps to take
    #: with the whole :attr:`Algorithm.buffer` when calling `step`.
    num_sgd_iter: int

    #: PPO hyperparameter indicating the minibatc size :attr:`Algorithm.buffer`
    #: is split into when updating the policy's model in :meth:`Algorithm.step`.
    #: It's usually best to maximize the minibatch size to reduce the variance
    #: associated with updating the policy's model, but also accelerate the
    #: computations when learning (assuming a CUDA device is being used).
    sgd_minibatch_size: int

    #: Whether to shuffle minibatches within :meth:`Algorithm.step`.
    #: Recommended, but not necessary if the minibatch size is large enough
    #: (e.g., the buffer is the batch).
    shuffle_minibatches: bool

    #: PPO hyperparameter similar to :attr:`Algorithm.clip_param` but for
    #: the value function estimate. A measure of max distance the model's
    #: value function is allowed to update away from previous value
    #: function samples.
    vf_clip_param: float

    #: Value function loss component weight. Only needs to be tuned
    #: when the policy and value function share parameters.
    vf_coeff: float

    def __post_init__(self) -> None:
        if not (0 < self.clip_param < 1):
            raise ValueError("`clip_param` must be in (0, 1).")

        if self.dual_clip_param is not None and not (self.dual_clip_param > 1):
            raise ValueError("`dual_clip_param` must be `None` or > 1.")

        if not (0 < self.gae_lambda <= 1):
            raise ValueError("`gae_lambda` must be in (0, 1].")

        if not (0 < self.gamma <= 1):
            raise ValueError("`gamma` must be in (0, 1].")

        if not (self.horizon > 0):
            raise ValueError("`horizon` must be > 0.")

        if self.horizons_per_env_reset == 0:
            raise ValueError("`horizons_per_env_reset` must be nonzero.")

        if not (self.max_grad_norm > 0):
            raise ValueError("`max_grad_norm` must be > 0.")

        if not (self.sgd_minibatch_size > 0):
            raise ValueError("`sgd_minibatch_size` must be > 0.")

        if not (self.vf_clip_param > 0):
            raise ValueError("`vf_clip_param` must be > 0.")

        if not (self.vf_coeff > 0):
            raise ValueError("`vf_coeff` must be > 0.")


@dataclass(frozen=True, kw_only=True)
class RecurrentAlgorithmHparams(AlgorithmHparams):
    """Recurrent PPO hyperparameters."""

    #: Truncated backpropagation through time sequence length.
    #: Not necessarily the sequence length the recurrent states
    #: are propagated for prior to being reset. This parameter
    #: coupled with :attr:`RecurrentAlgorithmHparams.seqs_per_state_reset`
    #: controls how many environment transitions are made before
    #: recurrent model states are reset or reinitialized.
    seq_len: int

    #: Number of sequences made within :meth:`RecurrentAlgorithmHparams.collect`
    #: before recurrent model states are reset or reinitialized. Recurrent
    #: model states are never reset or reinitialized if this parameter is
    #: negative.
    seqs_per_state_reset: int

    def __post_init__(self) -> None:
        if not (self.seq_len > 0):
            raise ValueError("`seq_len` must be > 0.")

        if self.seqs_per_state_reset == 0:
            raise ValueError("`seqs_per_state_reset` must be nonzero.")


@dataclass(kw_only=True)
class AlgorithmState:
    """Feedforward PPO state during training."""

    #: Flag indicating whether :meth:`Algorithm.collect` has been called
    #: at least once prior to calling :meth:`Algorithm.step`. Ensures
    #: dummy buffer data isn't used to update the policy.
    buffered: bool = False

    #: Number of times :meth:`Algorithm.collect` has been called.
    collect_calls: int = 0

    #: Running count of number of environment horizons sampled. This is
    #: equivalent to :attr:`Algorithm.collect_calls`. Used for tracking
    #: when to reset :attr:`Algorithm.env` based on
    #: :attr:`Algorithm.horizons_per_env_reset`.
    horizons: int = 0

    #: Number of times :meth:`Algorithm.step` has been called.
    step_calls: int = 0

    #: Total number of environment steps made.
    total_steps: int = 0


@dataclass(kw_only=True)
class RecurrentAlgorithmState(AlgorithmState):
    """Recurrent PPO state during training."""

    #: Number of recurrent sequences transitioned during training.
    seqs: int = 0


#: Values returned when collecting environment transitions.
CollectStats = TypedDict(
    "CollectStats",
    {
        "counting/collect_calls": int,
        "counting/horizons": int,
        "counting/total_steps": int,
        "profiling/collect_ms": float,
        "returns/min": float,
        "returns/max": float,
        "returns/mean": float,
        "returns/std": float,
        "rewards/min": float,
        "rewards/max": float,
        "rewards/mean": float,
        "rewards/std": float,
    },
    total=False,
)


#: Values returned when stepping/updating a policy.
StepStats = TypedDict(
    "StepStats",
    {
        "coefficients/entropy": float,
        "coefficients/vf": float,
        "counting/step_calls": int,
        "losses/entropy": float,
        "losses/policy": float,
        "losses/vf": float,
        "losses/total": float,
        "monitors/kl_div": float,
        "profiling/step_ms": float,
    },
    total=False,
)

#: Values returned during training.
class TrainStats(CollectStats, StepStats):
    ...


#: All the keys from :class:`TrainStats`.
TrainStatKey = Literal[
    "counting/collect_calls",
    "counting/horizons",
    "counting/total_steps",
    "profiling/collect_ms",
    "returns/min",
    "returns/max",
    "returns/mean",
    "returns/std",
    "rewards/min",
    "rewards/max",
    "rewards/mean",
    "rewards/std",
    "coefficients/entropy",
    "coefficients/vf",
    "counting/step_calls",
    "losses/entropy",
    "losses/policy",
    "losses/vf",
    "losses/total",
    "monitors/kl_div",
    "profiling/step_ms",
]
