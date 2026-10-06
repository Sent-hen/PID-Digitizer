"""Average precision from a ranked list of detections (used for edge mAP, where no
off-the-shelf implementation exists; symbol/node mAP use pycocotools instead)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def average_precision(scores: Sequence[float], is_tp: Sequence[bool], n_positives: int) -> float:
    """All-point interpolated AP (VOC 2010+ / area under the monotone PR curve).

    `n_positives` is the number of ground-truth positives (TP + FN). Returns NaN when
    there are no positives, so callers can exclude the class from the mean.
    """
    if n_positives == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    tp = np.asarray(is_tp, dtype=np.float64)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    # Tied scores form one threshold (as in sklearn): only evaluate the PR curve at the
    # last element of each tie group, so the result does not depend on input order.
    s_sorted = s[order]
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    ctp, cfp = ctp[last], cfp[last]
    recall = ctp / n_positives
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
