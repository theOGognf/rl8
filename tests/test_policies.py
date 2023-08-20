from tempfile import TemporaryDirectory
from typing import Iterator

import mlflow
import pytest
from tensordict import TensorDict

from rlstack import Policy, RecurrentPolicy
from rlstack.data import DataKeys
from rlstack.env import ContinuousDummyEnv, DiscreteDummyEnv, Env

NUM_ENVS = 64
HORIZON = 32
HORIZONS_PER_ENV_RESET = 2


@pytest.fixture
def tmpdir() -> Iterator[TemporaryDirectory]:
    with TemporaryDirectory() as tmp:
        yield tmp


@pytest.mark.parametrize("env_cls", [ContinuousDummyEnv, DiscreteDummyEnv])
def test_default_feedforward_policy_sample(env_cls: type[Env]) -> None:
    ENV = env_cls(1)
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


def test_feedforward_policy_save(tmpdir: TemporaryDirectory) -> None:
    ENV = DiscreteDummyEnv(NUM_ENVS)
    policy = Policy(ENV.observation_spec, ENV.action_spec)
    mlflow.pyfunc.save_model(
        f"{tmpdir}/model",
        python_model=policy.save(f"{tmpdir}/policy.pkl"),
        artifacts={"policy": f"{tmpdir}/policy.pkl"},
    )
    model = mlflow.pyfunc.load_model(f"{tmpdir}/model")
    obs = DiscreteDummyEnv(1).observation_spec.rand([1, 1]).cpu().numpy()
    df = model.predict({"obs": obs})
    assert {DataKeys.ACTIONS, DataKeys.LOGP, DataKeys.VALUES} == set(df.columns)


def test_recurrent_policy_save(tmpdir: TemporaryDirectory) -> None:
    ENV = DiscreteDummyEnv(NUM_ENVS)
    policy = RecurrentPolicy(ENV.observation_spec, ENV.action_spec)
    mlflow.pyfunc.save_model(
        f"{tmpdir}/model",
        python_model=policy.save(f"{tmpdir}/policy.pkl"),
        artifacts={"policy": f"{tmpdir}/policy.pkl"},
    )
    model = mlflow.pyfunc.load_model(f"{tmpdir}/model")
    obs = DiscreteDummyEnv(1).observation_spec.rand([1, 1]).cpu().numpy()
    df1, df2 = model.predict({"obs": obs})
    assert {
        DataKeys.ACTIONS,
        DataKeys.LOGP,
        DataKeys.VALUES,
    } == set(df1.columns)
    assert {DataKeys.HIDDEN_STATES, DataKeys.CELL_STATES} == set(df2.columns)
