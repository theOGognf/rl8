import math
from unittest.mock import patch

import pytest
import torch

from rl8 import Algorithm, RecurrentAlgorithm, RecurrentPolicy
from rl8.env import ContinuousDummyEnv, DiscreteDummyEnv, Env

NUM_ENVS = 64
HORIZON = 32
HORIZONS_PER_ENV_RESET = 2


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
