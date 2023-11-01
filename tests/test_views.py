import pytest
import torch
from tensordict import TensorDict

from rl8.data import DataKeys
from rl8.views import (
    PaddedRollingWindow,
    RollingWindow,
    ViewRequirement,
    pad_last_sequence,
    pad_whole_sequence,
    rolling_window,
)

B = 4
T = 1
SIZE = 2
TOTAL = B * T
INPUTS = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]]).float()
PADDING_MASK = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]]).bool()
PAD_LAST_SEQUENCE_CASE_0 = (
    torch.arange(B * T).reshape(B, T).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, SIZE],
    ),
    SIZE,
)

B = 2
T = 2
SIZE = 2
TOTAL = B * T * 2
INPUTS = torch.arange(TOTAL).reshape(B, T, 2).float()
PADDING_MASK = torch.zeros(B, SIZE).bool()
PAD_LAST_SEQUENCE_CASE_1 = (
    torch.arange(TOTAL).reshape(B, T, 2).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, SIZE],
    ),
    SIZE,
)

B = 2
T = 4
SIZE = 2
TOTAL = B * T
INPUTS = torch.arange(TOTAL).reshape(B, T, 1, 1, 1)[:, -SIZE:, ...].float()
PADDING_MASK = torch.zeros(B, SIZE).bool()
PAD_LAST_SEQUENCE_CASE_2 = (
    torch.arange(TOTAL).reshape(B, T, 1, 1, 1).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, SIZE],
    ),
    SIZE,
)

B = 2
T = 1
SIZE = 3
TOTAL = B * T
INPUTS = torch.cat(
    [
        torch.zeros(B, SIZE - T, 1, 1, 1),
        torch.arange(TOTAL).reshape(B, T, 1, 1, 1).float(),
    ],
    dim=1,
)
PADDING_MASK = torch.zeros(B, SIZE).bool()
PADDING_MASK[:, : SIZE - T] = True
PAD_LAST_SEQUENCE_CASE_3 = (
    torch.arange(TOTAL).reshape(B, T, 1, 1, 1).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, SIZE],
    ),
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        PAD_LAST_SEQUENCE_CASE_0,
        PAD_LAST_SEQUENCE_CASE_1,
        PAD_LAST_SEQUENCE_CASE_2,
        PAD_LAST_SEQUENCE_CASE_3,
    ],
)
def test_pad_last_sequence(
    inputs: torch.Tensor, expected: torch.Tensor | TensorDict, size: int
) -> None:
    assert (pad_last_sequence(inputs, size) == expected).all()


B = 4
T = 1
SIZE = 2
TOTAL = B * T
INPUTS = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]]).float()
PADDING_MASK = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]]).bool()
PAD_WHOLE_SEQUENCE_CASE_0 = (
    torch.arange(TOTAL).reshape(B, T).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, T + (SIZE - 1)],
    ),
    SIZE,
)

B = 2
T = 2
SIZE = 2
TOTAL = B * T * 2
INPUTS = torch.cat(
    [torch.zeros(B, (SIZE - 1), 2), torch.arange(TOTAL).reshape(B, T, 2)], dim=1
).float()
PADDING_MASK = torch.zeros(B, T + (SIZE - 1)).bool()
PADDING_MASK[:, : (SIZE - 1)] = True
PAD_WHOLE_SEQUENCE_CASE_1 = (
    torch.arange(TOTAL).reshape(B, T, 2).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, T + (SIZE - 1)],
    ),
    SIZE,
)

B = 2
T = 4
SIZE = 2
TOTAL = B * T
INPUTS = torch.cat(
    [torch.zeros(B, (SIZE - 1), 1, 1, 1), torch.arange(TOTAL).reshape(B, T, 1, 1, 1)],
    dim=1,
).float()
PADDING_MASK = torch.zeros(B, T + (SIZE - 1)).bool()
PADDING_MASK[:, : (SIZE - 1)] = True
PAD_WHOLE_SEQUENCE_CASE_2 = (
    torch.arange(TOTAL).reshape(B, T, 1, 1, 1).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, T + (SIZE - 1)],
    ),
    SIZE,
)


B = 2
T = 1
SIZE = 3
TOTAL = B * T
INPUTS = torch.cat(
    [torch.zeros(B, (SIZE - 1), 1, 1, 1), torch.arange(TOTAL).reshape(B, T, 1, 1, 1)],
    dim=1,
).float()
PADDING_MASK = torch.zeros(B, T + (SIZE - 1)).bool()
PADDING_MASK[:, : (SIZE - T)] = True
PAD_WHOLE_SEQUENCE_CASE_3 = (
    torch.arange(TOTAL).reshape(B, T, 1, 1, 1).float(),
    TensorDict(
        {
            DataKeys.INPUTS: INPUTS,
            DataKeys.PADDING_MASK: PADDING_MASK,
        },
        batch_size=[B, T + (SIZE - 1)],
    ),
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        PAD_WHOLE_SEQUENCE_CASE_0,
        PAD_WHOLE_SEQUENCE_CASE_1,
        PAD_WHOLE_SEQUENCE_CASE_2,
        PAD_WHOLE_SEQUENCE_CASE_3,
    ],
)
def test_pad_whole_sequence(
    inputs: torch.Tensor, expected: torch.Tensor | TensorDict, size: int
) -> None:
    assert (pad_whole_sequence(inputs, size) == expected).all()


B = 2
T = 1
SIZE = 2
TOTAL = B * T
INPUTS = TensorDict({"x": torch.arange(TOTAL).reshape(B, T).float()}, batch_size=[B, T])
EXPECTED = TensorDict({}, batch_size=[B, T + SIZE - 1])
EXPECTED["x"] = TensorDict({}, batch_size=[B, T + SIZE - 1])
EXPECTED["x"][DataKeys.INPUTS] = torch.cat(
    [torch.zeros(B, SIZE - 1), INPUTS["x"]], dim=1
)
EXPECTED["x"][DataKeys.PADDING_MASK] = torch.zeros(B, T + SIZE - 1).bool()
EXPECTED["x"][DataKeys.PADDING_MASK][:, : (SIZE - 1)] = True
PADDED_ROLLING_WINDOW_APPLY_ALL_CASE_0 = (
    INPUTS,
    RollingWindow.apply_all(EXPECTED, SIZE),
    SIZE,
)

B = 2
T = 4
SIZE = 2
TOTAL = B * T
INPUTS = TensorDict(
    {"x": torch.arange(TOTAL).reshape(B, T, 1).float()}, batch_size=[B, T]
)
EXPECTED = TensorDict({}, batch_size=[B, T + SIZE - 1])
EXPECTED["x"] = TensorDict({}, batch_size=[B, T + SIZE - 1])
EXPECTED["x"][DataKeys.INPUTS] = torch.cat(
    [torch.zeros(B, SIZE - 1, 1), INPUTS["x"]], dim=1
)
EXPECTED["x"][DataKeys.PADDING_MASK] = torch.zeros(B, T + SIZE - 1).bool()
EXPECTED["x"][DataKeys.PADDING_MASK][:, : (SIZE - 1)] = True
PADDED_ROLLING_WINDOW_APPLY_ALL_CASE_1 = (
    INPUTS,
    RollingWindow.apply_all(EXPECTED, SIZE),
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        PADDED_ROLLING_WINDOW_APPLY_ALL_CASE_0,
        PADDED_ROLLING_WINDOW_APPLY_ALL_CASE_1,
    ],
)
def test_padded_rolling_window_apply_all(
    inputs: TensorDict, expected: TensorDict, size: int
) -> None:
    assert (PaddedRollingWindow.apply_all(inputs, size) == expected).all()


B = 2
T = 1
SIZE = 2
TOTAL = B * T
INPUTS = TensorDict({"x": torch.arange(TOTAL).reshape(B, T).float()}, batch_size=[B, T])
EXPECTED = TensorDict({}, batch_size=[B, SIZE])
EXPECTED["x"] = TensorDict({}, batch_size=[B, SIZE])
EXPECTED["x"][DataKeys.INPUTS] = torch.cat(
    [torch.zeros(B, SIZE - 1), INPUTS["x"]], dim=1
)
EXPECTED["x"][DataKeys.PADDING_MASK] = torch.zeros(B, SIZE).bool()
EXPECTED["x"][DataKeys.PADDING_MASK][:, : (SIZE - 1)] = True
PADDED_ROLLING_WINDOW_APPLY_LAST_CASE_0 = (
    INPUTS,
    EXPECTED,
    SIZE,
)

B = 2
T = 4
SIZE = 2
TOTAL = B * T
INPUTS = TensorDict(
    {"x": torch.arange(TOTAL).reshape(B, T, 1).float()}, batch_size=[B, T]
)
EXPECTED = TensorDict({}, batch_size=[B, SIZE])
EXPECTED["x"] = TensorDict({}, batch_size=[B, SIZE])
EXPECTED["x"][DataKeys.INPUTS] = INPUTS["x"][:, -SIZE:, ...]
EXPECTED["x"][DataKeys.PADDING_MASK] = torch.zeros(B, SIZE).bool()
PADDED_ROLLING_WINDOW_APPLY_LAST_CASE_1 = (
    INPUTS,
    EXPECTED,
    SIZE,
)

B = 2
T = 1
SIZE = 3
TOTAL = B * T
INPUTS = TensorDict(
    {"x": torch.arange(TOTAL).reshape(B, T, 1).float()}, batch_size=[B, T], device="cpu"
)
EXPECTED = TensorDict({}, batch_size=[B, SIZE], device="cpu")
EXPECTED["x"] = TensorDict({}, batch_size=[B, SIZE])
EXPECTED["x"][DataKeys.INPUTS] = torch.cat(
    [torch.zeros(B, SIZE - T, 1), INPUTS["x"]], dim=1
)
EXPECTED["x"][DataKeys.PADDING_MASK] = torch.zeros(B, SIZE).bool()
EXPECTED["x"][DataKeys.PADDING_MASK][:, : SIZE - T, ...] = True
PADDED_ROLLING_WINDOW_APPLY_LAST_CASE_2 = (
    INPUTS,
    EXPECTED,
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        PADDED_ROLLING_WINDOW_APPLY_LAST_CASE_0,
        PADDED_ROLLING_WINDOW_APPLY_LAST_CASE_1,
        PADDED_ROLLING_WINDOW_APPLY_LAST_CASE_2,
    ],
)
def test_padded_rolling_window_apply_last(
    inputs: TensorDict, expected: TensorDict, size: int
) -> None:
    assert (PaddedRollingWindow.apply_last(inputs, size) == expected).all()


B = 2
T = 4
SIZE = 2
TOTAL = B * T
ROLLING_WINDOW_CASE_0 = (
    torch.arange(TOTAL).reshape(B, T).float(),
    torch.tensor([[[0, 1], [1, 2], [2, 3]], [[4, 5], [5, 6], [6, 7]]]).float(),
    SIZE,
)

B = 2
T = 4
SIZE = 2
TOTAL = B * T
ROLLING_WINDOW_CASE_1 = (
    torch.arange(TOTAL).reshape(B, T, 1).float(),
    torch.tensor(
        [[[[0], [1]], [[1], [2]], [[2], [3]]], [[[4], [5]], [[5], [6]], [[6], [7]]]]
    ).float(),
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        ROLLING_WINDOW_CASE_0,
        ROLLING_WINDOW_CASE_1,
    ],
)
def test_rolling_window(
    inputs: torch.Tensor, expected: torch.Tensor, size: int
) -> None:
    assert (rolling_window(inputs, size) == expected).all()


SIZE = 2
ROLLING_WINDOW_APPLY_ALL_CASE_0 = (
    ROLLING_WINDOW_CASE_0[0],
    ROLLING_WINDOW_CASE_0[1].reshape(-1, SIZE),
    SIZE,
)

SIZE = 2
ROLLING_WINDOW_APPLY_ALL_CASE_1 = (
    ROLLING_WINDOW_CASE_1[0],
    ROLLING_WINDOW_CASE_1[1].reshape(-1, SIZE, 1),
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        ROLLING_WINDOW_APPLY_ALL_CASE_0,
        ROLLING_WINDOW_APPLY_ALL_CASE_1,
    ],
)
def test_rolling_window_apply_all(
    inputs: torch.Tensor, expected: torch.Tensor, size: int
) -> None:
    assert (RollingWindow.apply_all(inputs, size) == expected).all()


B = 2
T = 4
SIZE = 2
TOTAL = B * T
INPUTS = TensorDict({"x": torch.arange(TOTAL).reshape(B, T).float()}, batch_size=[B, T])
ROLLING_WINDOW_APPLY_LAST_CASE_0 = (
    INPUTS,
    INPUTS[:, -SIZE:, ...],
    SIZE,
)

B = 2
T = 4
SIZE = 2
TOTAL = B * T
INPUTS = TensorDict(
    {"x": torch.arange(TOTAL).reshape(B, T, 1).float()}, batch_size=[B, T]
)
ROLLING_WINDOW_APPLY_LAST_CASE_1 = (
    INPUTS,
    INPUTS[:, -SIZE:, ...],
    SIZE,
)


@pytest.mark.parametrize(
    "inputs,expected,size",
    [
        ROLLING_WINDOW_APPLY_LAST_CASE_0,
        ROLLING_WINDOW_APPLY_LAST_CASE_1,
    ],
)
def test_rolling_window_apply_last(
    inputs: TensorDict, expected: TensorDict, size: int
) -> None:
    assert (RollingWindow.apply_last(inputs, size) == expected).all()


B = 20
T = 5
TOTAL = B * T
INPUTS = TensorDict({"x": torch.arange(TOTAL).reshape(B, T)}, batch_size=[B, T])
VIEW_REQUIREMENT_APPLY_ALL_CASE_0 = (INPUTS, INPUTS.reshape(-1))

B = 20
T = 5
TOTAL = B * T
INPUTS = TensorDict(
    {
        "x": TensorDict(
            {"y": torch.arange(TOTAL).reshape(B, T, 1, 1)}, batch_size=[B, T]
        )
    },
    batch_size=[B, T],
)
VIEW_REQUIREMENT_APPLY_ALL_CASE_1 = (INPUTS, INPUTS.reshape(-1))


@pytest.mark.parametrize(
    "inputs,expected",
    [
        VIEW_REQUIREMENT_APPLY_ALL_CASE_0,
        VIEW_REQUIREMENT_APPLY_ALL_CASE_1,
    ],
)
def test_view_requirement_apply_all(
    inputs: TensorDict, expected: torch.Tensor | TensorDict
) -> None:
    view_requirement = ViewRequirement(shift=0)
    out = {}
    out["x"] = view_requirement.apply_all("x", inputs)
    out_batch = TensorDict(out, batch_size=out["x"].size(0))
    assert (out_batch == expected).all()


B = 20
T = 5
TOTAL = B * T
INPUTS = TensorDict({"x": torch.arange(TOTAL).reshape(B, T)}, batch_size=[B, T])
VIEW_REQUIREMENT_APPLY_LAST_CASE_0 = (INPUTS, INPUTS[:, -1, ...])

B = 20
T = 5
TOTAL = B * T
INPUTS = TensorDict(
    {
        "x": TensorDict(
            {"y": torch.arange(TOTAL).reshape(B, T, 1, 1)}, batch_size=[B, T]
        )
    },
    batch_size=[B, T],
)
VIEW_REQUIREMENT_APPLY_LAST_CASE_1 = (INPUTS, INPUTS[:, -1, ...])


@pytest.mark.parametrize(
    "inputs,expected",
    [
        VIEW_REQUIREMENT_APPLY_LAST_CASE_0,
        VIEW_REQUIREMENT_APPLY_LAST_CASE_1,
    ],
)
def test_view_requirement_apply_last(
    inputs: TensorDict, expected: torch.Tensor | TensorDict
) -> None:
    view_requirement = ViewRequirement(shift=0)
    out = {}
    out["x"] = view_requirement.apply_last("x", inputs)
    out_batch = TensorDict(out, batch_size=out["x"].size(0))
    assert (out_batch == expected).all()
