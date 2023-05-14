"""Definitions related to data passed between algorithm modules."""

from typing import Literal, TypedDict

import torch

Device = str | torch.device


class DataKeys:
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

    STATES = "states"

    HIDDEN_STATES = "hidden_states"

    CELL_STATES = "cell_states"


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
