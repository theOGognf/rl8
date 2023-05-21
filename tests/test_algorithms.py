import math

import pytest
import torch

from rlstack import Algorithm, RecurrentAlgorithm
from rlstack.env import ContinuousDummyEnv, DiscreteDummyEnv, Env


@pytest.mark.parametrize("algorithm_cls", [Algorithm, RecurrentAlgorithm])
@pytest.mark.parametrize("env_cls", [ContinuousDummyEnv, DiscreteDummyEnv])
def test_algorithm(algorithm_cls: type[Algorithm], env_cls: type[Env]) -> None:
    SEED = 42
    NUM_ENVS = 64
    HORIZON = 8
    ENTROPY_COEFF = 1e-2
    RTOL = 1e-3
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
