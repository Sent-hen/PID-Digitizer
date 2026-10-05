from __future__ import annotations

import numpy as np
import pytest

from pid.config import load_config
from pid.detect.augment import augment, degrade_tile, hflip, repeat_factors, rot90


def tile_with_box() -> tuple[np.ndarray, np.ndarray]:
    img = np.full((64, 64), 255, np.uint8)
    img[10:20, 30:50] = 0  # box x 30..50, y 10..20
    return img, np.array([[30.0, 10.0, 50.0, 20.0]])


def ink_box(img: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(img < 128)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rot90_moves_boxes_with_ink(k: int) -> None:
    img, b = tile_with_box()
    out, ob = rot90(img, b, k)
    np.testing.assert_allclose(ob[0], ink_box(out))


def test_hflip_moves_boxes_with_ink() -> None:
    img, b = tile_with_box()
    out, ob = hflip(img, b)
    np.testing.assert_allclose(ob[0], ink_box(out))


def test_degrade_tile_keeps_box_on_content() -> None:
    img = np.full((512, 512), 255, np.uint8)
    img[200:260, 220:300] = 0
    b = np.array([[220.0, 200.0, 300.0, 260.0]])
    out, ob, lab = degrade_tile(img, b, np.array([3]), 3, load_config("degrade"), "k")
    assert len(ob) == 1 and lab[0] == 3
    ys, xs = np.nonzero(out[150:310, 170:350] < 128)
    cx, cy = xs.mean() + 170, ys.mean() + 150
    assert ob[0, 0] <= cx <= ob[0, 2] and ob[0, 1] <= cy <= ob[0, 3]


def test_augment_deterministic_for_seed() -> None:
    img, b = tile_with_box()
    cfg = {"rot90": True, "flip": True, "degrade_prob": 1.0}
    dcfg = load_config("degrade")
    a = augment(img, b, np.array([1]), np.random.default_rng(7), cfg, dcfg)
    c = augment(img, b, np.array([1]), np.random.default_rng(7), cfg, dcfg)
    np.testing.assert_array_equal(a[0], c[0])
    np.testing.assert_array_equal(a[1], c[1])


def test_repeat_factors_upweight_rare_class() -> None:
    # class 4 appears in 1 of 100 tiles, class 0 in all
    sets = [[0]] * 99 + [[0, 4]]
    r = repeat_factors(sets, threshold=0.04)
    assert r[0] == 1.0
    assert r[-1] == pytest.approx(np.sqrt(0.04 / 0.01))  # = 2.0
    assert repeat_factors([[], [1]], 0.04)[0] == 1.0  # empty tiles weight 1
