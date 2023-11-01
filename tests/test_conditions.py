from rl8.conditions import (
    And,
    HitsLowerBound,
    HitsUpperBound,
    Plateaus,
    StopsDecreasing,
    StopsIncreasing,
)


def test_and() -> None:
    and_ = And(
        [
            HitsUpperBound("returns/mean", 100.0),
            HitsUpperBound("counting/step_calls", 50.0),
        ]
    )
    assert not and_({"returns/mean": 100.0, "counting/step_calls": 1})
    assert and_({"returns/mean": 100.0, "counting/step_calls": 50})


def test_hits_lower_bound() -> None:
    hits_lower_bound = HitsLowerBound("returns/mean", -100.0)
    assert not hits_lower_bound({"returns/mean": 1})
    assert hits_lower_bound({"returns/mean": -200.0})


def test_hits_lower_bound() -> None:
    hits_upper_bound = HitsUpperBound("returns/mean", 100.0)
    assert not hits_upper_bound({"returns/mean": 1})
    assert hits_upper_bound({"returns/mean": 200.0})


def test_plateaus() -> None:
    plateaus = Plateaus("returns/mean", patience=2, rtol=2e-1)
    assert not plateaus({"returns/mean": 1})
    assert not plateaus({"returns/mean": 0.9})
    assert plateaus({"returns/mean": 1})


def test_stops_decreasing() -> None:
    stops_decreasing = StopsDecreasing("returns/mean", patience=2)
    assert not stops_decreasing({"returns/mean": 1})
    assert not stops_decreasing({"returns/mean": 1.1})
    assert stops_decreasing({"returns/mean": 1.2})


def test_stops_increasing() -> None:
    stops_increasing = StopsIncreasing("returns/mean", patience=2)
    assert not stops_increasing({"returns/mean": 1})
    assert not stops_increasing({"returns/mean": 0.9})
    assert stops_increasing({"returns/mean": 0.8})
