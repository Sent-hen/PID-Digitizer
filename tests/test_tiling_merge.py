from __future__ import annotations

import numpy as np
import pytest

from pid.detect.merge import merge_detections
from pid.io.drawing import BBox, Drawing, Symbol
from pid.preprocess.tiling import (
    Tile,
    boxes_to_global,
    boxes_to_local,
    crop_tile,
    make_tiles,
    rescale_factor,
    scale_drawing,
    touches_tile_border,
)


@pytest.mark.parametrize(("w", "h"), [(7000, 4500), (1024, 1024), (500, 300), (3000, 1025)])
def test_tiles_cover_image_and_stay_inside(w: int, h: int) -> None:
    tiles = make_tiles(w, h, 1024, 256)
    cover = np.zeros((h, w), bool)
    for t in tiles:
        cover[t.y0:t.y1, t.x0:t.x1] = True
        assert t.x0 >= 0 and t.y0 >= 0
        assert t.x1 <= max(w, 1024) and t.y1 <= max(h, 1024)
    assert cover.all()


def test_tile_overlap_at_least_requested() -> None:
    xs = sorted({t.x0 for t in make_tiles(7000, 4500, 1024, 256)})
    assert all(b - a <= 1024 - 256 for a, b in zip(xs, xs[1:], strict=False))


def test_crop_and_coordinate_round_trip() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (900, 1300), dtype=np.uint8)
    boxes = np.array([[10.0, 20.0, 50.0, 60.0], [1250.5, 850.25, 1299.0, 899.0]])
    for t in make_tiles(1300, 900, 512, 128):
        crop = crop_tile(img, t)
        assert crop.shape == (512, 512)
        h, w = min(t.y1, 900) - t.y0, min(t.x1, 1300) - t.x0
        np.testing.assert_array_equal(crop[:h, :w], img[t.y0:t.y0 + h, t.x0:t.x0 + w])
        np.testing.assert_allclose(boxes_to_global(boxes_to_local(boxes, t), t), boxes,
                                   atol=1e-9)


def test_crop_pads_small_image() -> None:
    img = np.zeros((100, 200), np.uint8)
    crop = crop_tile(img, Tile(0, 0, 256))
    assert crop.shape == (256, 256)
    assert crop[:100, :200].max() == 0 and crop[150, 150] == 255


def test_touches_interior_border_only() -> None:
    t = Tile(768, 0, 1024)
    b = np.array([[768.0, 10, 800, 40],    # touches left (interior) edge
                  [900.0, 0, 950, 30],     # touches top, but that is the image border
                  [900.0, 100, 950, 150]])  # fully inside
    np.testing.assert_array_equal(touches_tile_border(b, t, 4000, 3000), [True, False, False])


def test_scale_drawing_round_trip_within_1px() -> None:
    d = Drawing(id="x", width=2604, height=1744, symbols=[
        Symbol(id="a", cls="valve", bbox=BBox(xmin=101.3, ymin=55.5, xmax=120.2, ymax=77.7))])
    f = rescale_factor(2604, 1744, 7000)
    back = scale_drawing(scale_drawing(d, f), 1 / f)
    np.testing.assert_allclose(back.symbols[0].bbox.as_xyxy(), d.symbols[0].bbox.as_xyxy(),
                               atol=1e-6)
    assert abs(back.width - 2604) <= 1 and abs(back.height - 1744) <= 1  # type: ignore[operator]


# --- merging ------------------------------------------------------------------------

def _merge(boxes: list[list[float]], scores: list[float], labels: list[int],
           trunc: list[bool] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return merge_detections(np.array(boxes, float), np.array(scores, float),
                            np.array(labels, np.int64),
                            None if trunc is None else np.array(trunc, bool))


def test_nms_removes_duplicate_same_class() -> None:
    b, s, _ = _merge([[0, 0, 10, 10], [1, 0, 11, 10]], [0.9, 0.8], [0, 0])
    assert len(b) == 1 and s[0] == 0.9


def test_nms_is_class_aware() -> None:
    b, _, labels = _merge([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 1])
    assert sorted(labels.tolist()) == [0, 1]


def test_object_spanning_tile_border_full_copy_wins() -> None:
    # Symbol at x 1000..1060. Tile A (0..1024) sees a truncated fragment with a higher
    # score; tile B (768..1792) sees all of it. IoU(fragment, full) = 24/60 < 0.5.
    full, frag = [1000, 50, 1060, 90], [1000, 50, 1024, 90]
    b, _, _ = _merge([frag, full], [0.95, 0.7], [2, 2], trunc=[True, False])
    assert len(b) == 1
    np.testing.assert_allclose(b[0], full)


def test_large_object_truncated_in_both_tiles_is_unioned() -> None:
    left, right = [900, 0, 1024, 50], [990, 0, 1300, 50]
    b, _, _ = _merge([left, right], [0.8, 0.9], [1, 1], trunc=[True, True])
    # IoS = 34*50 / (124*50) = 0.27 -> not merged at default thresholds; with overlap
    # inside the smaller box they are:
    assert len(b) == 2
    inner = [1000, 0, 1024, 50]
    b2, _, _ = _merge([inner, right], [0.8, 0.9], [1, 1], trunc=[True, True])
    assert len(b2) == 1
    np.testing.assert_allclose(b2[0], [990, 0, 1300, 50])


def test_adjacent_distinct_symbols_survive() -> None:
    b, _, _ = _merge([[0, 0, 10, 10], [10, 0, 20, 10]], [0.9, 0.9], [0, 0])
    assert len(b) == 2


def test_empty_input() -> None:
    b, s, labels = _merge([], [], [])
    assert b.shape == (0, 4) and len(s) == 0 and len(labels) == 0
