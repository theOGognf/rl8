import math
from tempfile import TemporaryDirectory
from typing import Iterator
from unittest.mock import patch

import mlflow
import pytest
import torch

from rlstack import (
    Algorithm,
    MLflowPolicyModel,
    MLflowRecurrentPolicyModel,
    RecurrentAlgorithm,
    RecurrentPolicy,
)
from rlstack.data import DataKeys
from rlstack.env import ContinuousDummyEnv, DiscreteDummyEnv, Env

NUM_ENVS = 64
HORIZON = 32
HORIZONS_PER_ENV_RESET = 2


@pytest.fixture
def tmpdir() -> Iterator[TemporaryDirectory]:
    with TemporaryDirectory() as tmp:
        yield tmp


@pytest.mark.parametrize("algorithm_cls", [Algorithm, RecurrentAlgorithm])
@pytest.mark.parametrize("env_cls", [ContinuousDummyEnv, DiscreteDummyEnv])
def test_algorithm(algorithm_cls: type[Algorithm], env_cls: type[Env]) -> None:
    SEED = 42
    ENTROPY_COEFF = 1e-2
    RTOL = 1e-5
    torch.manual_seed(SEED)
    algo = algorithm_cls(
        env_cls, num_envs=NUM_ENVS, horizon=HORIZON, entropy_coeff=ENTROPY_COEFF
    )
    algo.collect()
    step_stats_non_accumulated = algo.step()

    torch.manual_seed(SEED)
    algo = algorithm_cls(
        env_cls,
        num_envs=NUM_ENVS,
        horizon=HORIZON,
        accumulate_grads=True,
        entropy_coeff=ENTROPY_COEFF,
        sgd_minibatch_size=NUM_ENVS,
    )
    algo.collect()
    step_stats_accumulated = algo.step()

    assert math.isclose(
        step_stats_non_accumulated["losses/entropy"],
        step_stats_accumulated["losses/entropy"],
        rel_tol=RTOL,
    )
    assert math.isclose(
        step_stats_non_accumulated["losses/policy"],
        step_stats_accumulated["losses/policy"],
        rel_tol=RTOL,
    )
    assert math.isclose(
        step_stats_non_accumulated["losses/total"],
        step_stats_accumulated["losses/total"],
        rel_tol=RTOL,
    )
    assert math.isclose(
        step_stats_non_accumulated["losses/vf"],
        step_stats_accumulated["losses/vf"],
        rel_tol=RTOL,
    )
    assert math.isclose(
        step_stats_non_accumulated["monitors/kl_div"],
        step_stats_accumulated["monitors/kl_div"],
        rel_tol=RTOL,
    )


def test_feedforward_algorithm_resets() -> None:
    algo = Algorithm(
        DiscreteDummyEnv,
        horizon=HORIZON,
        num_envs=NUM_ENVS,
        horizons_per_env_reset=HORIZONS_PER_ENV_RESET,
    )
    with (patch.object(DiscreteDummyEnv, "reset", wraps=algo.env.reset) as reset,):
        algo.collect()
        assert algo.state.horizons == 1
        assert reset.call_count == 1
        algo.collect()
        assert algo.state.horizons == 2
        assert reset.call_count == 1
        algo.collect()
        assert algo.state.horizons == 3
        assert reset.call_count == 2


def test_feedforward_algorithm_save_policy(tmpdir: TemporaryDirectory) -> None:
    algo = Algorithm(
        DiscreteDummyEnv,
        horizon=HORIZON,
        num_envs=NUM_ENVS,
        horizons_per_env_reset=HORIZONS_PER_ENV_RESET,
    )
    algo.save_policy(f"{tmpdir}/policy.pkl")
    mlflow.pyfunc.save_model(
        f"{tmpdir}/model",
        python_model=MLflowPolicyModel(),
        artifacts={"policy": f"{tmpdir}/policy.pkl"},
    )
    model = mlflow.pyfunc.load_model(f"{tmpdir}/model")
    obs = DiscreteDummyEnv(1).observation_spec.rand([1, 1]).cpu().numpy()
    df = model.predict({"obs": obs})
    assert {DataKeys.ACTIONS, DataKeys.LOGP, DataKeys.VALUES} == set(df.columns)


def test_recurrent_algorithm_resets() -> None:
    algo = RecurrentAlgorithm(
        DiscreteDummyEnv,
        horizon=HORIZON,
        num_envs=NUM_ENVS,
        seq_len=4,
        seqs_per_state_reset=8,
    )
    with (
        patch.object(DiscreteDummyEnv, "reset", wraps=algo.env.reset) as reset,
        patch.object(
            RecurrentPolicy, "init_states", wraps=algo.policy.init_states
        ) as init_states,
    ):
        algo.collect()
        assert algo.state.horizons == 1
        assert reset.call_count == 1
        assert algo.state.seqs == 8
        assert init_states.call_count == 1
        algo.collect()
        assert algo.state.horizons == 2
        assert reset.call_count == 2
        assert algo.state.seqs == 16
        assert init_states.call_count == 2


def test_recurrent_algorithm_save_policy(tmpdir: TemporaryDirectory) -> None:
    algo = RecurrentAlgorithm(
        DiscreteDummyEnv, horizon=32, num_envs=64, seq_len=4, seqs_per_state_reset=8
    )
    algo.save_policy(f"{tmpdir}/policy.pkl")
    mlflow.pyfunc.save_model(
        f"{tmpdir}/model",
        python_model=MLflowRecurrentPolicyModel(),
        artifacts={"policy": f"{tmpdir}/policy.pkl"},
    )
    model = mlflow.pyfunc.load_model(f"{tmpdir}/model")
    obs = DiscreteDummyEnv(1).observation_spec.rand([1, 1]).cpu().numpy()
    df = model.predict({"obs": obs})
    assert {
        DataKeys.ACTIONS,
        DataKeys.LOGP,
        DataKeys.HIDDEN_STATES,
        DataKeys.CELL_STATES,
        DataKeys.VALUES,
    } == set(df.columns)
