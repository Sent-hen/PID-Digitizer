"""Training-time augmentation and class-balanced sampling (numpy only, CI-testable).

* 90-degree rotations and flips: label-preserving here, because none of the 7
  PID2Graph symbol classes encodes direction (an arrow pointing left and one pointing
  up are both `arrow`; valves are drawn in either orientation on real sheets).
* Scan degradation: the Milestone 2 simulator applied to a tile, with boxes mapped
  through its skew; boxes pushed mostly outside the tile are dropped.
* Repeat-factor sampling (Gupta et al., LVIS 2019): a tile's sampling weight is
  max over its classes of max(1, sqrt(t / f_c)), f_c = fraction of tiles with class c.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pid.preprocess.degrade import degrade

U8 = NDArray[np.uint8]
F = NDArray[np.float64]
I64 = NDArray[np.int64]


def rot90(img: U8, boxes: F, k: int) -> tuple[U8, F]:
    """Rotate a square tile counter-clockwise by k*90 degrees, with its boxes."""
    s = img.shape[0]
    out_b = boxes.copy()
    for _ in range(k % 4):
        x0, y0, x1, y1 = out_b[:, 0].copy(), out_b[:, 1].copy(), out_b[:, 2].copy(), out_b[:, 3]
        # (x, y) -> (y, s - x) for np.rot90 (counter-clockwise)
        out_b = np.stack([y0, s - x1, y1, s - x0], axis=1)
    return np.ascontiguousarray(np.rot90(img, k)), out_b


def hflip(img: U8, boxes: F) -> tuple[U8, F]:
    s = img.shape[1]
    out = boxes.copy()
    out[:, 0], out[:, 2] = s - boxes[:, 2], s - boxes[:, 0]
    return np.ascontiguousarray(img[:, ::-1]), out


def degrade_tile(img: U8, boxes: F, labels: I64, level: int, degcfg: Mapping[str, Any],
                 key: str, min_visible: float = 0.5) -> tuple[U8, F, I64]:
    out, M = degrade(img, level, degcfg, key)
    if len(boxes) == 0:
        return out, boxes, labels
    xs = np.stack([boxes[:, 0], boxes[:, 2], boxes[:, 2], boxes[:, 0]], axis=1)
    ys = np.stack([boxes[:, 1], boxes[:, 1], boxes[:, 3], boxes[:, 3]], axis=1)
    tx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
    ty = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
    hb = np.stack([tx.min(1), ty.min(1), tx.max(1), ty.max(1)], axis=1)
    s = float(img.shape[0])
    cb = np.clip(hb, 0.0, s)
    area = (hb[:, 2] - hb[:, 0]) * (hb[:, 3] - hb[:, 1])
    vis = (cb[:, 2] - cb[:, 0]).clip(0) * (cb[:, 3] - cb[:, 1]).clip(0)
    keep = vis >= min_visible * np.maximum(area, 1e-9)
    return out, cb[keep], labels[keep]


def augment(img: U8, boxes: F, labels: I64, rng: np.random.Generator,
            cfg: Mapping[str, Any], degcfg: Mapping[str, Any]) -> tuple[U8, F, I64]:
    if cfg.get("rot90"):
        img, boxes = rot90(img, boxes, int(rng.integers(0, 4)))
    if cfg.get("flip") and rng.random() < 0.5:
        img, boxes = hflip(img, boxes)
    p = float(cfg.get("degrade_prob", 0.0))
    if p > 0 and rng.random() < p:
        level = int(rng.integers(1, 4))
        img, boxes, labels = degrade_tile(img, boxes, labels, level, degcfg,
                                          key=f"aug{int(rng.integers(1 << 31))}")
    return img, boxes, labels


def repeat_factors(label_sets: Sequence[Sequence[int]], threshold: float) -> list[float]:
    """Per-tile sampling weights for repeat-factor sampling."""
    n = max(1, len(label_sets))
    freq = Counter(c for labs in label_sets for c in set(labs))
    rc = {c: max(1.0, float(np.sqrt(threshold / (k / n)))) for c, k in freq.items()}
    return [max((rc[c] for c in set(labs)), default=1.0) for labs in label_sets]
