"""Fixed-size overlapping tiling with coordinate mapping, and sheet rescaling.

Coordinates are (x, y) in pixels of the *working* image (after `rescale`). The last
tile in each row/column is aligned to the image edge, so every pixel is covered and
no tile extends past the image except when the image is smaller than one tile
(then the tile is padded on read).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from pid.io.drawing import BBox, Drawing

U8 = NDArray[np.uint8]


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size


def _starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size, stride))
    starts.append(length - size)
    return sorted(set(starts))


def make_tiles(width: int, height: int, size: int, overlap: int) -> list[Tile]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    stride = size - overlap
    return [Tile(x, y, size) for y in _starts(height, size, stride)
            for x in _starts(width, size, stride)]


def crop_tile(image: U8, tile: Tile, fill: int = 255) -> U8:
    """Crop a tile from a HxW (or HxWxC) image, padding with `fill` past the edges."""
    h, w = image.shape[:2]
    out = np.full((tile.size, tile.size, *image.shape[2:]), fill, dtype=image.dtype)
    ys, xs = slice(tile.y0, min(tile.y1, h)), slice(tile.x0, min(tile.x1, w))
    out[: ys.stop - ys.start, : xs.stop - xs.start] = image[ys, xs]
    return out


def boxes_to_global(boxes: NDArray[np.float64], tile: Tile) -> NDArray[np.float64]:
    """Map Nx4 xyxy boxes from tile coordinates to working-image coordinates."""
    out: NDArray[np.float64] = boxes + np.array([tile.x0, tile.y0, tile.x0, tile.y0], float)
    return out


def boxes_to_local(boxes: NDArray[np.float64], tile: Tile) -> NDArray[np.float64]:
    out: NDArray[np.float64] = boxes - np.array([tile.x0, tile.y0, tile.x0, tile.y0], float)
    return out


def touches_tile_border(boxes: NDArray[np.float64], tile: Tile, width: int, height: int,
                        margin: float = 2.0) -> NDArray[np.bool_]:
    """True for (global) boxes that touch an *interior* tile edge, i.e. are likely cut
    off by the tile; edges that coincide with the image border do not count."""
    b = boxes
    left = (b[:, 0] <= tile.x0 + margin) & (tile.x0 > 0)
    top = (b[:, 1] <= tile.y0 + margin) & (tile.y0 > 0)
    right = (b[:, 2] >= tile.x1 - margin) & (tile.x1 < width)
    bottom = (b[:, 3] >= tile.y1 - margin) & (tile.y1 < height)
    out: NDArray[np.bool_] = left | top | right | bottom
    return out


# --- rescaling -------------------------------------------------------------------

def rescale_factor(width: int, height: int, target_long_side: int | None) -> float:
    return 1.0 if not target_long_side else target_long_side / max(width, height)


def rescale_image(image: U8, factor: float) -> U8:
    if abs(factor - 1.0) < 1e-9:
        return image
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    out = cast(U8, cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=interp))
    return out


def scale_drawing(drawing: Drawing, factor: float) -> Drawing:
    """Return a copy with all geometry multiplied by `factor`."""
    d = drawing.model_copy(deep=True)

    def sb(b: BBox) -> BBox:
        return BBox(xmin=b.xmin * factor, ymin=b.ymin * factor,
                    xmax=b.xmax * factor, ymax=b.ymax * factor)

    for s in d.symbols:
        s.bbox = sb(s.bbox)
    d.ignore_regions = [sb(r) for r in d.ignore_regions]
    for t in d.texts:
        t.bbox = sb(t.bbox)
    if d.width and d.height:
        d.width, d.height = round(d.width * factor), round(d.height * factor)
    return d
