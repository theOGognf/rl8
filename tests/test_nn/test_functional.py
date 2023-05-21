import torch
from tensordict import TensorDict

from rlstack.data import DataKeys
from rlstack.nn.functional import (
    generalized_advantage_estimate,
    mask_from_lengths,
    masked_avg,
    masked_categorical_sample,
    masked_max,
)


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


def test_masked_avg() -> None:
    x = torch.arange(4).reshape(2, 2, 1).float()
    mask = torch.ones(4).reshape(2, 2).float()
    mask[1, :] = 0.0
    avg = masked_avg(x, mask=mask, dim=0)
    assert (avg == x[0, :]).all()


def test_masked_categorical_sample() -> None:
    x = torch.arange(4).reshape(2, 2, 1).float()
    mask = torch.ones(4).reshape(2, 2).float()
    mask[:, 1] = 0.0
    logits, samples = masked_categorical_sample(x, mask=mask, dim=1)
    assert (logits == x[:, 0:1]).all()
    assert (samples.flatten() == torch.zeros(4)).all()


def test_mask_from_lengths() -> None:
    x = torch.arange(4).reshape(2, 2, 1).float()
    lengths = torch.ones(2)
    mask = mask_from_lengths(x, lengths)
    max_, argmax = masked_max(x, mask=mask, dim=1)
    assert (max_ == x[:, 0:1]).all()
    assert (argmax.flatten() == torch.zeros(2)).all()


def test_masked_max() -> None:
    x = torch.arange(4).reshape(2, 2, 1).float()
    mask = torch.ones(4).reshape(2, 2).float()
    mask[1, :] = 0.0
    max_, argmax = masked_max(x, mask=mask, dim=0)
    assert (max_ == x[0, :]).all()
    assert (argmax.flatten() == torch.zeros(2)).all()
