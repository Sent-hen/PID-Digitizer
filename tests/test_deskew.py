from __future__ import annotations

import cv2
import numpy as np
import pytest

from pid.config import load_config
from pid.preprocess.deskew import estimate_skew, invert_affine, rotate


def pid_like(w: int = 2400, h: int = 1600) -> np.ndarray:
    img = np.full((h, w), 255, np.uint8)
    for y in range(200, h - 100, 230):
        cv2.line(img, (100, y), (w - 100, y), 0, 5)
    for x in range(300, w - 100, 400):
        cv2.line(img, (x, 150), (x, h - 150), 0, 5)
    cv2.putText(img, "TEXT 1234", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    return img


@pytest.mark.parametrize("skew", [0.0, 0.4, -0.9, 1.5])
def test_estimates_and_undoes_skew(skew: float) -> None:
    cfg = load_config("pipeline")["deskew"]
    skewed, _ = rotate(pid_like(), skew)
    angle = estimate_skew(skewed, cfg)
    assert angle == pytest.approx(-skew, abs=0.06)


def test_inverse_round_trip() -> None:
    _, M = rotate(pid_like(), 1.2)
    Mi = invert_affine(M)
    p = np.array([700.0, 300.0, 1.0])
    q = M @ p
    np.testing.assert_allclose(Mi @ np.append(q, 1.0), p[:2], atol=1e-9)
