"""Merging per-tile detections into one set of full-sheet detections.

Class-aware greedy NMS with two extensions for tiled inference:

* Boxes cut off by an interior tile edge (`truncated`) rank after complete boxes, so
  the complete copy of an object from a neighbouring tile wins.
* Overlap is measured as max(IoU, IoS) where IoS = intersection / smaller area: a
  truncated fragment lies almost entirely inside the full box (high IoS) even though
  its IoU with it is low.
* When two *truncated* boxes of the same object are merged (an object larger than the
  tile overlap, cut in every tile), the kept box grows to their union.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float64]


def _overlap(box: F, others: F) -> tuple[F, F]:
    ix0 = np.maximum(box[0], others[:, 0])
    iy0 = np.maximum(box[1], others[:, 1])
    ix1 = np.minimum(box[2], others[:, 2])
    iy1 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = a + b - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    ios = np.where(np.minimum(a, b) > 0, inter / np.maximum(np.minimum(a, b), 1e-9), 0.0)
    return iou, ios


def merge_detections(boxes: F, scores: F, labels: NDArray[np.int64],
                     truncated: NDArray[np.bool_] | None = None, iou_thr: float = 0.5,
                     ios_thr: float = 0.8) -> tuple[F, F, NDArray[np.int64]]:
    """Return merged (boxes, scores, labels). Inputs are global xyxy boxes."""
    n = len(boxes)
    if n == 0:
        return boxes.reshape(0, 4), scores, labels
    trunc = np.zeros(n, bool) if truncated is None else truncated
    out_b, out_s, out_l = [], [], []
    for cls in np.unique(labels):
        idx = np.nonzero(labels == cls)[0]
        # complete boxes first, then by descending score
        order = idx[np.lexsort((-scores[idx], trunc[idx].astype(int)))]
        alive = np.ones(len(order), bool)
        for i in range(len(order)):
            if not alive[i]:
                continue
            k = order[i]
            keep = boxes[k].copy()
            rest = np.nonzero(alive[i + 1:])[0] + i + 1
            if len(rest):
                iou, ios = _overlap(keep, boxes[order[rest]])
                dup = (iou >= iou_thr) | (ios >= ios_thr)
                for j in rest[dup]:
                    if trunc[k] and trunc[order[j]]:
                        b = boxes[order[j]]
                        keep = np.array([min(keep[0], b[0]), min(keep[1], b[1]),
                                         max(keep[2], b[2]), max(keep[3], b[3])])
                alive[rest[dup]] = False
            out_b.append(keep)
            out_s.append(scores[k])
            out_l.append(cls)
    return np.array(out_b, float), np.array(out_s, float), np.array(out_l, np.int64)
