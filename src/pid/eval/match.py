"""Matching predicted symbols to ground-truth symbols."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from pid.io.drawing import Symbol

FloatArray = NDArray[np.float64]


def boxes_array(symbols: Sequence[Symbol]) -> FloatArray:
    if not symbols:
        return np.zeros((0, 4))
    return np.array([s.bbox.as_xyxy() for s in symbols], dtype=np.float64)


def iou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU between xyxy boxes, shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    out: FloatArray = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    return out


def giou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise generalized IoU (Rezatofighi et al., 2019), in [-1, 1]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iou = iou_matrix(a, b)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    cx0 = np.minimum(a[:, None, 0], b[None, :, 0])
    cy0 = np.minimum(a[:, None, 1], b[None, :, 1])
    cx1 = np.maximum(a[:, None, 2], b[None, :, 2])
    cy1 = np.maximum(a[:, None, 3], b[None, :, 3])
    hull = (cx1 - cx0) * (cy1 - cy0)
    out: FloatArray = iou - np.where(hull > 0, (hull - union) / np.where(hull > 0, hull, 1), 0)
    return out


def greedy_match(gt: Sequence[Symbol], pred: Sequence[Symbol], iou_thr: float = 0.5,
                 class_aware: bool = False) -> dict[str, str]:
    """COCO-style matching: predictions in descending score order take the best
    still-unmatched GT box with IoU >= `iou_thr`. Returns {pred_id: gt_id}."""
    iou = iou_matrix(boxes_array(pred), boxes_array(gt))
    if class_aware:
        same = np.array([[p.cls == g.cls for g in gt] for p in pred], dtype=bool)
        iou = np.where(same.reshape(iou.shape), iou, 0.0)
    order = sorted(range(len(pred)), key=lambda i: -pred[i].score)
    taken: set[int] = set()
    out: dict[str, str] = {}
    for i in order:
        if iou.shape[1] == 0:
            break
        cand = iou[i].copy()
        if taken:
            cand[list(taken)] = -1.0
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            taken.add(j)
            out[pred[i].id] = gt[j].id
    return out


def hungarian_match(gt: Sequence[Symbol], pred: Sequence[Symbol],
                    min_iou: float | None = None) -> dict[str, str]:
    """One-to-one assignment maximizing total gIoU (as in the PID2Graph edge mAP).

    The paper applies no IoU threshold after assignment; `min_iou` optionally drops
    assigned pairs whose IoU is below it. Returns {pred_id: gt_id}.
    """
    if not gt or not pred:
        return {}
    a, b = boxes_array(pred), boxes_array(gt)
    rows, cols = linear_sum_assignment(-giou_matrix(a, b))
    iou = iou_matrix(a, b) if min_iou is not None else None
    out: dict[str, str] = {}
    for r, c in zip(rows, cols, strict=True):
        if iou is not None and min_iou is not None and iou[r, c] < min_iou:
            continue
        out[pred[r].id] = gt[c].id
    return out
