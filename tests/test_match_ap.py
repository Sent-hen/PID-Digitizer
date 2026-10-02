from __future__ import annotations

import math

import numpy as np
import pytest

from pid.eval.ap import average_precision
from pid.eval.match import giou_matrix, greedy_match, hungarian_match, iou_matrix
from pid.io.drawing import BBox, Symbol


def sym(i: str, x: float, y: float, s: float = 10, score: float = 1.0, cls: str = "valve",
        ) -> Symbol:
    return Symbol(id=i, cls=cls, bbox=BBox(xmin=x, ymin=y, xmax=x + s, ymax=y + s), score=score)


def test_iou_known_values() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_giou_disjoint_negative() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[20, 0, 30, 10]], dtype=float)
    # hull area 300, union 200 -> giou = 0 - 100/300
    np.testing.assert_allclose(giou_matrix(a, b), [[-1 / 3]])


def test_greedy_prefers_high_score_and_is_one_to_one() -> None:
    gt = [sym("g", 0, 0)]
    pred = [sym("lo", 0, 0, score=0.2), sym("hi", 1, 0, score=0.9)]
    assert greedy_match(gt, pred) == {"hi": "g"}


def test_greedy_threshold_and_class_aware() -> None:
    gt = [sym("g", 0, 0, cls="valve")]
    assert greedy_match(gt, [sym("p", 6, 0)]) == {}  # IoU 4/16 < 0.5
    assert greedy_match(gt, [sym("p", 0, 0, cls="pump")], class_aware=True) == {}
    assert greedy_match(gt, [sym("p", 0, 0, cls="pump")]) == {"p": "g"}


def test_hungarian_global_optimum() -> None:
    gt = [sym("g1", 0, 0), sym("g2", 8, 0)]
    pred = [sym("p1", 4, 0), sym("p2", 9, 0)]
    assert hungarian_match(gt, pred) == {"p1": "g1", "p2": "g2"}
    assert hungarian_match(gt, pred, min_iou=0.5) == {"p2": "g2"}


def test_ap_perfect_and_empty() -> None:
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert average_precision([], [], 3) == 0.0
    assert math.isnan(average_precision([0.5], [False], 0))


def test_ap_known_value() -> None:
    # ranks: TP, FP, TP with 4 positives -> recall 0.25@P1, 0.5@P(2/3)
    ap = average_precision([0.9, 0.8, 0.7], [True, False, True], 4)
    assert ap == pytest.approx(0.25 * 1.0 + 0.25 * (2 / 3))
