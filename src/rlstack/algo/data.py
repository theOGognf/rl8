"""Definitions related to data passed between algorithm modules."""

from typing import TypedDict

import torch

DEVICE = int | str | torch.device


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

    #: Key denoting entropy of a probability distribution (a measure of a
    #: probability distribution's randomness) loss.
    ENTROPY_LOSS = "losses/entropy"

    #: Key denoting KL divergence (a measure of distance between two probability
    #: distributions) loss.
    KL_DIV_LOSS = "losses/kl_div"

    #: Key denoting loss associated with a learning algorithm's policy loss.
    #: For PPO, this is a clipped policy loss ratio weighted by advantages.
    POLICY_LOSS = "losses/policy"

    #: Key denoting loss associated with a policy's model's ability to predict
    #: values from the "values" key.
    VF_LOSS = "losses/vf"

    #: Key denoting sum of all losses.
    TOTAL_LOSS = "losses/total"

    #: Key denoting entropy coefficient for the entropy loss.
    ENTROPY_COEFF = "coefficients/entropy"

    #: Key denoting KL divergence coefficient for the KL divergence loss.
    KL_DIV_COEFF = "coefficients/kl_div"

    #: Key denoting value function loss coefficient.
    VF_COEFF = "coefficients/vf"


#: Stats updated and tracked within `Algorithm.collect`.
CollectStats = TypedDict(
    "CollectStats",
    {
        "profiling/collect_ms": float,
        "rewards/min": float,
        "rewards/max": float,
        "rewards/mean": float,
        "rewards/std": float,
    },
    total=False,
)


#: Stats updated and tracked within `Algorithm.step`.
StepStats = TypedDict(
    "StepStats",
    {
        "coefficients/entropy": float,
        "coefficients/kl_div": float,
        "coefficients/vf": float,
        "losses/entropy": float,
        "losses/kl_div": float,
        "losses/policy": float,
        "losses/vf": float,
        "losses/total": float,
        "profiling/step_ms": float,
    },
    total=False,
)
