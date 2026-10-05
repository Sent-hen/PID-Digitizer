"""End-to-end walking-skeleton pipeline: image -> symbols -> lines -> graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from pid.graph.assemble import assemble
from pid.io.drawing import BBox, Drawing, Symbol
from pid.lines.extract import extract_segments
from pid.preprocess.crop import detect_content_box
from pid.preprocess.tiling import rescale_factor, rescale_image, scale_drawing

U8 = NDArray[np.uint8]
# A detector maps a working-scale grayscale image to symbols in working coordinates.
Detector = Callable[[U8], list[Symbol]]

STRUCTURAL = ("crossing", "ankle")


def load_gray(path: str | Path) -> U8:
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"cannot decode image {path}")
    out = cast(U8, img)
    return out


def digitize(image_path: str | Path, cfg: Mapping[str, Any], drawing_id: str = "",
             subset: str = "", detector: Detector | None = None,
             oracle_symbols: Sequence[Symbol] | None = None,
             gray: U8 | None = None) -> Drawing:
    """Run the pipeline on one sheet. Exactly one of `detector` / `oracle_symbols`
    (ground-truth symbols in original coordinates, for error-propagation analysis)
    must be given. Returns a `Drawing` in original image coordinates."""
    if (detector is None) == (oracle_symbols is None):
        raise ValueError("give exactly one of detector / oracle_symbols")
    if gray is None:  # `gray` lets callers pass a pre-processed (e.g. degraded) image
        gray = load_gray(image_path)
    H, W = gray.shape
    f = rescale_factor(W, H, cfg["preprocess"].get("target_long_side"))
    work = rescale_image(gray, f)

    crop_cfg = cfg.get("crop", {})
    box = detect_content_box(work, crop_cfg) if crop_cfg.get("enabled") else None
    if box is not None:
        # Blank the frame/title-block area so it yields no line segments.
        x0, y0 = max(0, int(box.xmin)), max(0, int(box.ymin))
        x1, y1 = int(np.ceil(box.xmax)), int(np.ceil(box.ymax))
        masked = np.full_like(work, 255)
        masked[y0:y1, x0:x1] = work[y0:y1, x0:x1]
    else:
        masked = work

    if oracle_symbols is not None:
        tmp = Drawing(id=drawing_id, symbols=[s for s in oracle_symbols
                                              if s.cls not in STRUCTURAL])
        symbols = scale_drawing(tmp, f).symbols
    else:
        assert detector is not None
        symbols = detector(work)
        if box is not None:  # drop detections centered in the frame/title block
            symbols = [sy for sy in symbols
                       if box.xmin <= sy.bbox.center[0] <= box.xmax
                       and box.ymin <= sy.bbox.center[1] <= box.ymax]

    segments = extract_segments(masked, [s.bbox for s in symbols], cfg["lines"])
    graph = assemble(drawing_id, symbols, segments, cfg["graph"], subset)
    out = scale_drawing(graph, 1.0 / f)
    # Structural nodes are fixed-size markers in original pixels (dataset convention),
    # not physical extents: re-center them at their size instead of scaling.
    half = float(cfg["graph"].get("struct_size", 8)) / 2
    for s in out.symbols:
        if s.cls in STRUCTURAL:
            cx, cy = s.bbox.center
            s.bbox = BBox(xmin=cx - half, ymin=cy - half, xmax=cx + half, ymax=cy + half)
    out.width, out.height, out.image_path = W, H, str(image_path)
    return out
