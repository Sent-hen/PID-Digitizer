from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from pid.config import load_config
from pid.io.drawing import BBox, Drawing, Symbol
from pid.preprocess.degrade import degrade, transform_drawing


@pytest.fixture(scope="module")
def dcfg() -> dict[str, Any]:
    return load_config("degrade")


def sheet() -> np.ndarray:
    img = np.full((600, 900), 255, np.uint8)
    cv2.rectangle(img, (400, 250), (480, 330), 0, -1)  # a solid symbol
    cv2.line(img, (50, 100), (850, 100), 0, 5)
    return img


def test_level0_is_identity(dcfg: dict[str, Any]) -> None:
    img = sheet()
    out, M = degrade(img, 0, dcfg, "x/1")
    assert out is img
    np.testing.assert_allclose(M, [[1, 0, 0], [0, 1, 0]])


@pytest.mark.parametrize("level", [1, 2, 3])
def test_deterministic_for_fixed_seed(dcfg: dict[str, Any], level: int) -> None:
    a, ma = degrade(sheet(), level, dcfg, "x/1")
    b, mb = degrade(sheet(), level, dcfg, "x/1")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ma, mb)
    assert a.shape == sheet().shape and a.dtype == np.uint8


def test_different_sheets_get_different_randomness(dcfg: dict[str, Any]) -> None:
    a, _ = degrade(sheet(), 2, dcfg, "x/1")
    b, _ = degrade(sheet(), 2, dcfg, "x/2")
    assert not np.array_equal(a, b)


def test_severity_increases_damage(dcfg: dict[str, Any]) -> None:
    # Damage = pixels whose ink/paper status flips (mean gray error is misleading:
    # level 3 binarizes, which restores pure-white paper).
    img = sheet()
    ink = img < 128
    errs = [int(((degrade(img, lv, dcfg, "x/1")[0] < 128) ^ ink).sum()) for lv in (1, 2, 3)]
    assert errs[0] < errs[1] < errs[2]


def test_level3_is_binary(dcfg: dict[str, Any]) -> None:
    out, _ = degrade(sheet(), 3, dcfg, "x/1")
    assert set(np.unique(out).tolist()) <= {0, 255}


@pytest.mark.parametrize("level", [1, 2, 3])
def test_transform_tracks_skewed_content(dcfg: dict[str, Any], level: int) -> None:
    img = sheet()
    out, M = degrade(img, level, dcfg, "x/1")
    d = Drawing(id="1", symbols=[
        Symbol(id="s", cls="valve", bbox=BBox(xmin=400, ymin=250, xmax=480, ymax=330)),
        Symbol(id="a", cls="ankle", bbox=BBox(xmin=396, ymin=286, xmax=404, ymax=294))])
    t = transform_drawing(d, M)
    b = t.symbols[0].bbox
    # the symbol's ink centroid in the degraded image lies inside the mapped box
    ys, xs = np.nonzero(out[200:380, 350:530] < 100)
    cx, cy = xs.mean() + 350, ys.mean() + 200
    assert b.xmin <= cx <= b.xmax and b.ymin <= cy <= b.ymax
    assert abs(cx - b.center[0]) < 3 and abs(cy - b.center[1]) < 3
    assert t.symbols[1].bbox.width == pytest.approx(8.0)  # structural size preserved


def test_crop_finds_frame_and_title_column() -> None:
    from pid.preprocess.crop import detect_content_box

    img = np.full((1200, 1800), 255, np.uint8)
    cv2.rectangle(img, (40, 30), (1760, 1170), 0, 4)   # frame
    cv2.line(img, (1400, 30), (1400, 1170), 0, 4)       # notes/title column separator
    cv2.line(img, (200, 600), (1200, 600), 0, 5)        # a pipe (must not be a frame)
    cfg = load_config("pipeline")["crop"]
    b = detect_content_box(img, cfg)
    assert b is not None
    assert abs(b.xmin - 40) <= 10 and abs(b.ymin - 30) <= 10
    assert abs(b.xmax - 1400) <= 10 and abs(b.ymax - 1170) <= 10


def test_crop_none_without_frame() -> None:
    from pid.preprocess.crop import detect_content_box

    img = np.full((1200, 1800), 255, np.uint8)
    cv2.line(img, (100, 600), (1700, 600), 0, 5)   # long pipe, no frame
    cv2.line(img, (900, 100), (900, 1100), 0, 5)
    assert detect_content_box(img, load_config("pipeline")["crop"]) is None
