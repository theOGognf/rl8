import pytest
from tensordict import TensorDict

from rlstack import Policy, RecurrentPolicy
from rlstack.data import DataKeys
from rlstack.env import ContinuousDummyEnv, DiscreteDummyEnv, Env


@pytest.mark.parametrize("env_cls", [ContinuousDummyEnv, DiscreteDummyEnv])
def test_default_feedforward_policy_sample(env_cls: type[Env]) -> None:
    ENV = env_cls(1)
    NUM_ENVS = 10
    HORIZON = 5
    INPUT_BATCH = TensorDict(
        {DataKeys.OBS: ENV.observation_spec.rand([NUM_ENVS, HORIZON])},
        batch_size=[NUM_ENVS, HORIZON],
    )
    policy = Policy(ENV.observation_spec, ENV.action_spec)
    out = policy.sample(
        INPUT_BATCH,
        kind="last",
        inplace=False,
        requires_grad=False,
        return_actions=True,
        return_logp=True,
        return_values=True,
        return_views=True,
    )
    assert out is not INPUT_BATCH
    assert out[DataKeys.FEATURES].batch_size == (NUM_ENVS,)
    assert out[DataKeys.ACTIONS].shape == (NUM_ENVS, 1)
    assert out[DataKeys.LOGP].shape == (NUM_ENVS, 1)
    assert out[DataKeys.VALUES].shape == (NUM_ENVS, 1)
    assert out[DataKeys.VIEWS].batch_size == (NUM_ENVS,)

    out = policy.sample(
        INPUT_BATCH,
        kind="all",
        inplace=False,
        requires_grad=False,
        return_actions=True,
        return_logp=True,
        return_values=True,
        return_views=True,
    )
    assert out is not INPUT_BATCH
    assert out[DataKeys.FEATURES].batch_size == (NUM_ENVS * HORIZON,)
    assert out[DataKeys.ACTIONS].shape == (NUM_ENVS * HORIZON, 1)
    assert out[DataKeys.LOGP].shape == (NUM_ENVS * HORIZON, 1)
    assert out[DataKeys.VALUES].shape == (NUM_ENVS * HORIZON, 1)
    assert out[DataKeys.VIEWS].batch_size == (NUM_ENVS * HORIZON,)


@pytest.mark.parametrize("env_cls", [ContinuousDummyEnv, DiscreteDummyEnv])
def test_default_recurrent_policy_sample(env_cls: type[Env]) -> None:
    ENV = env_cls(1)
    NUM_ENVS = 10
    HORIZON = 5
    INPUT_BATCH = TensorDict(
        {DataKeys.OBS: ENV.observation_spec.rand([NUM_ENVS, HORIZON])},
        batch_size=[NUM_ENVS, HORIZON],
    )
    policy = RecurrentPolicy(ENV.observation_spec, ENV.action_spec)
    out, _ = policy.sample(
        INPUT_BATCH[:, -1:, ...],
        inplace=False,
        requires_grad=False,
        return_actions=True,
        return_logp=True,
        return_values=True,
    )
    assert out is not INPUT_BATCH
    assert out[DataKeys.FEATURES].batch_size == (NUM_ENVS,)
    assert out[DataKeys.ACTIONS].shape == (NUM_ENVS, 1)
    assert out[DataKeys.LOGP].shape == (NUM_ENVS, 1)
    assert out[DataKeys.VALUES].shape == (NUM_ENVS, 1)

    out, _ = policy.sample(
        INPUT_BATCH,
        inplace=False,
        requires_grad=False,
        return_actions=True,
        return_logp=True,
        return_values=True,
    )
    assert out is not INPUT_BATCH
    assert out[DataKeys.FEATURES].batch_size == (NUM_ENVS * HORIZON,)
    assert out[DataKeys.ACTIONS].shape == (NUM_ENVS * HORIZON, 1)
    assert out[DataKeys.LOGP].shape == (NUM_ENVS * HORIZON, 1)
    assert out[DataKeys.VALUES].shape == (NUM_ENVS * HORIZON, 1)
