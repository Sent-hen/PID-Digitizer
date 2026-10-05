"""Skew estimation and correction (Milestone 2).

The line stage assumes axis-aligned pipes; at 0.8 deg of skew a 7000 px pipe drifts
~100 px and edge F1 collapses (RESULTS R-006). P&IDs are dominated by long horizontal
and vertical lines, so the skew angle is the rotation that makes the horizontal
projection profile sharpest (maximum variance of row ink counts). Coarse-to-fine
search on a block-minimum downscaled ink mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
F = NDArray[np.float64]


def _small_ink(gray: U8, factor: int, threshold: int) -> NDArray[np.float32]:
    H, W = gray.shape
    f = max(1, factor)
    small = gray[: H - H % f, : W - W % f].reshape(H // f, f, W // f, f).min(axis=(1, 3))
    return (small < threshold).astype(np.float32)


def _score(ink: NDArray[np.float32], angle: float) -> float:
    h, w = ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return float(np.var(rot.sum(axis=1)))


def estimate_skew(gray: U8, cfg: Mapping[str, Any]) -> float:
    """Angle (degrees, OpenCV convention) that rotates the sheet back to level."""
    ink = _small_ink(gray, int(cfg["factor"]), int(cfg["threshold"]))
    max_a = float(cfg["max_angle"])
    coarse = np.arange(-max_a, max_a + 1e-9, float(cfg["coarse_step"]))
    best = max(coarse, key=lambda a: _score(ink, float(a)))
    step = float(cfg["coarse_step"])
    fine = np.arange(best - step, best + step + 1e-9, float(cfg["fine_step"]))
    return float(max(fine, key=lambda a: _score(ink, float(a))))


def rotate(gray: U8, angle: float, fill: int = 255) -> tuple[U8, F]:
    """Rotate about the image center; returns (image, 2x3 affine old -> new)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return cast(U8, out), M.astype(np.float64)


def invert_affine(M: F) -> F:
    out: F = cv2.invertAffineTransform(M).astype(np.float64)
    return out
