"""Orthogonal line-segment extraction (walking-skeleton version).

P&ID pipes and signal lines in PID2Graph are axis-aligned (every GT edge between
structural nodes in Dataset-PID/Synthetic is exactly horizontal or vertical), so lines
are recovered with directional morphology instead of a generic skeleton:

1. binarize, and blank out detected symbol boxes (lines then end at symbol borders);
2. per direction: close with a short kernel (bridges dash gaps), open with a long
   kernel (keeps only runs >= `min_length`, removing text and symbol remnants);
3. each connected component thinner than `max_width` becomes one segment, whose
   centerline is the component centroid;
4. line style: the fraction of the segment covered by ink *before* closing (dashed
   lines have gaps) - below `solid_min_fill` it is non-solid.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from pid.io.drawing import BBox, LineType

U8 = NDArray[np.uint8]


@dataclass
class Segment:
    orient: Literal["h", "v"]
    c: float  # centerline: y for horizontal, x for vertical
    a0: float  # start along the line: x for horizontal, y for vertical
    a1: float
    width: float
    fill: float = 1.0
    line_type: LineType = LineType.SOLID

    @property
    def length(self) -> float:
        return self.a1 - self.a0

    def point(self, a: float) -> tuple[float, float]:
        return (a, self.c) if self.orient == "h" else (self.c, a)


def binarize(gray: U8, threshold: int) -> U8:
    out: U8 = (gray < threshold).astype(np.uint8)
    return out


def mask_boxes(binary: U8, boxes: Iterable[BBox], pad: int = 1) -> U8:
    out = binary.copy()
    h, w = out.shape
    for b in boxes:
        x0, y0 = max(0, int(np.floor(b.xmin)) - pad), max(0, int(np.floor(b.ymin)) - pad)
        x1, y1 = min(w, int(np.ceil(b.xmax)) + pad + 1), min(h, int(np.ceil(b.ymax)) + pad + 1)
        out[y0:y1, x0:x1] = 0
    return out


def _segments_one_direction(binary: U8, orient: Literal["h", "v"], cfg: Mapping[str, Any],
                            ) -> list[Segment]:
    L, gap, max_w = int(cfg["min_length"]), int(cfg["bridge_gap"]), float(cfg["max_width"])
    if orient == "h":
        k_close, k_open = (gap, 1), (L, 1)  # cv2 kernel size is (width, height)
    else:
        k_close, k_open = (1, gap), (1, L)
    m = binary
    if gap > 1:
        m = cast(U8, cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, k_close)))
    m = cast(U8, cv2.morphologyEx(m, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, k_open)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = binary.shape
    max_len = float(cfg.get("max_length_frac", 1.0)) * (W if orient == "h" else H)
    ctx_off, ctx_w = int(cfg.get("context_offset", 4)), int(cfg.get("context_width", 8))
    max_ctx = float(cfg.get("max_context_ink", 1.0))
    # Work on the transposed image for vertical lines so both cases share code.
    img = binary if orient == "h" else binary.T
    segs: list[Segment] = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        cx, cy = cent[i]
        if orient == "h":
            a0, along, thick, c = x, w, h, cy
        else:
            a0, along, thick, c = y, h, w, cx
        if thick > max_w or along < L or along > max_len:
            continue
        ci, ht = int(round(c)), thick // 2
        band = img[max(0, ci - ht - 1): ci + ht + 2, a0:a0 + along]
        cover = band.any(axis=0)
        # Ink in thin bands just beside the line: lines run through empty space, text
        # strokes merged by closing do not.
        lo = img[max(0, ci - ht - ctx_off - ctx_w): max(0, ci - ht - ctx_off), a0:a0 + along]
        hi = img[ci + ht + ctx_off + 1: ci + ht + ctx_off + 1 + ctx_w, a0:a0 + along]
        side = np.zeros(along, bool)
        if lo.size:
            side |= lo.any(axis=0)
        if hi.size:
            side |= hi.any(axis=0)
        if side.mean() > max_ctx:
            continue
        seg = Segment(orient, float(c), float(a0), float(a0 + along - 1), float(thick))
        seg.fill = float(cover.mean()) if cover.size else 0.0
        seg.line_type = (LineType.SOLID if seg.fill >= float(cfg["solid_min_fill"])
                         else LineType.NON_SOLID)
        segs.append(seg)
    return segs


def extract_segments(gray: U8, symbol_boxes: Iterable[BBox], cfg: Mapping[str, Any],
                     ) -> list[Segment]:
    binary = mask_boxes(binarize(gray, int(cfg["threshold"])), symbol_boxes,
                        int(cfg.get("mask_pad", 1)))
    return (_segments_one_direction(binary, "h", cfg)
            + _segments_one_direction(binary, "v", cfg))
