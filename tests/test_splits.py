from __future__ import annotations

import pytest

from pid.io.splits import rare_class_aware_pick


def counts() -> dict[str, dict[str, int]]:
    # pump only in s/1 and s/2; tank in s/1, s/3, s/4
    return {
        "s/0": {"valve": 5},
        "s/1": {"valve": 2, "pump": 3, "tank": 1},
        "s/2": {"valve": 1, "pump": 2},
        "s/3": {"valve": 4, "tank": 2},
        "s/4": {"valve": 3, "tank": 1},
        "s/5": {"valve": 2},
    }


def test_constraints_hold() -> None:
    c = counts()
    train = rare_class_aware_pick(c, 2, ["pump", "tank"], seed=0)
    test = [u for u in c if u not in train]
    assert len(train) == 2
    for cls in ("pump", "tank", "valve"):
        assert sum(c[u].get(cls, 0) for u in test) >= 1  # every class stays in test
    for cls in ("pump", "tank"):
        assert sum(c[u].get(cls, 0) for u in train) >= 1  # rare classes in train


def test_deterministic() -> None:
    a = rare_class_aware_pick(counts(), 2, ["pump", "tank"], seed=3)
    b = rare_class_aware_pick(counts(), 2, ["pump", "tank"], seed=3)
    assert a == b


def test_infeasible_raises() -> None:
    with pytest.raises(ValueError):
        rare_class_aware_pick({"s/0": {"pump": 1}, "s/1": {"valve": 1}}, 1, ["pump"], seed=0)
