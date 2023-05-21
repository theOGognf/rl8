import torch
from tensordict import TensorDict

from rlstack.data import DataKeys
from rlstack.nn.functional import generalized_advantage_estimate


def test_generalized_advantage_estimate() -> None:
    NUM_ENVS = 10
    HORIZON = 5
    INPUT_BATCH = TensorDict(
        {
            DataKeys.REWARDS: torch.ones(NUM_ENVS, HORIZON + 1, 1),
            DataKeys.VALUES: torch.ones(NUM_ENVS, HORIZON + 1, 1),
        },
        batch_size=[NUM_ENVS, HORIZON + 1],
    )
    UNDISCOUNTED_RETURNS = torch.flip(
        torch.cumsum(INPUT_BATCH[DataKeys.REWARDS], dim=1), dims=(1,)
    )
    out = generalized_advantage_estimate(
        INPUT_BATCH,
        gae_lambda=1,
        gamma=1,
        inplace=False,
        normalize=False,
        return_returns=True,
    )
    assert out is not INPUT_BATCH
    assert (out[DataKeys.ADVANTAGES] == (UNDISCOUNTED_RETURNS - 1)).all()
    assert (out[DataKeys.RETURNS] == UNDISCOUNTED_RETURNS).all()

    out = generalized_advantage_estimate(
        INPUT_BATCH,
        gae_lambda=1,
        gamma=1,
        inplace=True,
        normalize=False,
        return_returns=True,
    )
    assert out is INPUT_BATCH
    assert (out[DataKeys.ADVANTAGES] == (UNDISCOUNTED_RETURNS - 1)).all()
    assert (out[DataKeys.RETURNS] == UNDISCOUNTED_RETURNS).all()
