"""Frame and title-block cropping (Milestone 2).

The dataset's ignore regions (frame margins, the notes/title-block column) leave one
*content box* where the diagram lives. `detect_content_box` estimates it from long
ruling lines: the drawing frame is the outermost pair of long horizontal and vertical
lines, and a title-block/notes column or strip is a long interior line near the right
or bottom edge. Sheets without a frame (Synthetic) return None (no crop).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from pid.io.drawing import BBox, Drawing

U8 = NDArray[np.uint8]


def _long_lines(binary: U8, orient: str, min_frac: float) -> list[tuple[float, int, int]]:
    """Long straight rules as (position, start, end) in the given (downscaled) image."""
    h, w = binary.shape
    if orient == "h":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(w * min_frac)), 1))
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(h * min_frac))))
    m = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if orient == "h":
            out.append((float(cent[i][1]), int(x), int(x + bw)))
        else:
            out.append((float(cent[i][0]), int(y), int(y + bh)))
    return out


def detect_content_box(gray: U8, cfg: Mapping[str, Any]) -> BBox | None:
    """Content box in the coordinates of `gray`, or None if no frame is found."""
    # Block-minimum downscale: area averaging fades 1-2 px rules below the ink
    # threshold, the minimum keeps them.
    f = max(1, round(1.0 / float(cfg["scale"])))
    H, W = gray.shape
    small = gray[: H - H % f, : W - W % f].reshape(H // f, f, W // f, f).min(axis=(1, 3))
    s = 1.0 / f
    binary = (small < int(cfg["threshold"])).astype(np.uint8)
    h, w = binary.shape
    frame_frac = float(cfg["frame_min_frac"])
    hs = _long_lines(binary, "h", frame_frac)
    vs = _long_lines(binary, "v", frame_frac)
    if len(hs) < 2 or len(vs) < 2:
        return None
    top = min(p for p, _, _ in hs)
    bottom = max(p for p, _, _ in hs)
    left = min(p for p, _, _ in vs)
    right = max(p for p, _, _ in vs)
    # A drawing frame hugs the sheet edges (at least 3 sides: OPEN100 puts the title
    # block outside the frame on the right) and its top/bottom rules span its width;
    # long pipes on frameless sheets (Synthetic) do neither.
    edge = float(cfg["frame_edge_frac"])
    near = [top <= edge * h, bottom >= (1 - edge) * h, left <= edge * w,
            right >= (1 - edge) * w]
    if sum(near) < 3:
        return None
    def coverage(y: float) -> float:  # union length of collinear rule pieces at y
        cover = np.zeros(w, bool)
        for p, a0, a1 in hs:
            if abs(p - y) < 3:
                cover[max(0, a0):min(w, a1)] = True
        return float(cover[int(left):int(right)].mean()) if right > left else 0.0

    if coverage(top) < 0.9 or coverage(bottom) < 0.9:
        return None

    # Interior separators: long rules spanning most of the frame, near the right/bottom.
    sep_frac = float(cfg["separator_min_frac"])
    band = float(cfg["separator_band"])
    margin = float(cfg["frame_margin_frac"]) * w
    for p, _, _ in _long_lines(binary, "v", sep_frac * (bottom - top) / h):
        if left + margin < p < right - margin and p > right - band * (right - left):
            right = min(right, p)
    for p, a0, a1 in _long_lines(binary, "h", sep_frac * (right - left) / w):
        # a bottom strip separator must cut across the full content width
        if (top + margin < p < bottom - margin and p > bottom - band * (bottom - top)
                and a0 <= left + margin and a1 >= right - margin):
            bottom = min(bottom, p)
    return BBox(xmin=left / s, ymin=top / s, xmax=right / s, ymax=bottom / s)


def gt_content_box(d: Drawing, grid: int = 8) -> BBox | None:
    """Bounding box of the area not covered by the annotated ignore regions."""
    if not d.ignore_regions or not d.width or not d.height:
        return None
    gw, gh = -(-d.width // grid), -(-d.height // grid)  # cells covering the image only
    mask = np.zeros((gh, gw), bool)
    for r in d.ignore_regions:
        x0, y0 = max(0, int(r.xmin // grid)), max(0, int(r.ymin // grid))
        x1, y1 = int(np.ceil(r.xmax / grid)), int(np.ceil(r.ymax / grid))
        mask[y0:y1, x0:x1] = True
    ys, xs = np.nonzero(~mask)
    if not len(xs):
        return None
    return BBox(xmin=float(np.min(xs) * grid), ymin=float(np.min(ys) * grid),
                xmax=float((np.max(xs) + 1) * grid), ymax=float((np.max(ys) + 1) * grid))


def box_iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    iy = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
