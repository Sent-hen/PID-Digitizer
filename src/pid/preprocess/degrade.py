"""Reproducible scan-degradation simulator (Milestone 2).

PID2Graph sheets are clean digital renders; real legacy P&IDs are scans or faxes.
`degrade` applies a fixed composite of contrast loss, blur, skew, noise, resolution
loss, JPEG compression and (at the top level) fax-like binarization, with parameters
from `configs/degrade.yaml`. Randomness (skew angle, noise) comes only from seeds
derived from (config seed, drawing uid, level), so output is deterministic. The skew
direction and relative size are fixed per sheet and scaled by level.

Skew moves content, so the affine transform is returned and `transform_drawing`
maps ground truth into the degraded image's coordinates.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from pid.io.drawing import BBox, Drawing

U8 = NDArray[np.uint8]
F = NDArray[np.float64]
STRUCTURAL = ("crossing", "ankle")


def rng_for(seed: int, uid: str, level: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(uid.encode()), level])


def degrade(gray: U8, level: int, cfg: Mapping[str, Any], uid: str = "",
            ) -> tuple[U8, F]:
    """Return (degraded image, 2x3 affine mapping original -> degraded pixels).
    Level 0 returns the input unchanged with the identity transform."""
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if level == 0:
        return gray, ident
    p = cfg["levels"][level]
    rng = rng_for(int(cfg["seed"]), uid, level)
    h, w = gray.shape
    img: NDArray[Any] = gray.astype(np.float32)

    # contrast loss: ink lightens toward the paper, paper darkens toward gray
    bg = float(p["background"])
    img = bg - (bg - img * (bg / 255.0)) * float(p["contrast"])

    if float(p["blur_sigma"]) > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(p["blur_sigma"]))

    # One skew direction and relative magnitude per sheet (shared across levels), so a
    # sheet is skewed more at every higher level and robustness curves are monotone.
    srng = rng_for(int(cfg["seed"]), uid, 0)
    sign = 1.0 if srng.random() < 0.5 else -1.0
    angle = sign * float(srng.uniform(0.5, 1.0)) * float(p["skew_deg"])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=bg)

    if float(p["noise_sigma"]) > 0:
        img = img + rng.normal(0.0, float(p["noise_sigma"]), img.shape).astype(np.float32)
    sp = float(p["speckle"])
    if sp > 0:
        r = rng.random(img.shape, dtype=np.float32)
        img[r < sp / 2] = 0.0
        img[(r >= sp / 2) & (r < sp)] = 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    ds = float(p["downscale"])
    if ds < 1.0:
        small = cv2.resize(img, (max(1, round(w * ds)), max(1, round(h * ds))),
                           interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    q = int(p["jpeg_quality"])
    if q < 100:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if dec is None:
            raise RuntimeError("JPEG decoding failed")
        img = dec

    if p.get("binarize"):
        img = np.where(img < int(p["binarize_threshold"]), 0, 255).astype(np.uint8)
    return cast(U8, img), M.astype(np.float64)


def _apply(M: F, x: float, y: float) -> tuple[float, float]:
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])


def transform_drawing(d: Drawing, M: F, struct_size: float = 8.0) -> Drawing:
    """Map ground truth through the affine `M`.

    Symbol boxes become the axis-aligned hull of their rotated corners (what an
    annotator would draw on the skewed scan); structural nodes keep their fixed size
    and are re-centered, matching the dataset convention.
    """
    out = d.model_copy(deep=True)

    def hull(b: BBox) -> BBox:
        pts = [_apply(M, x, y) for x in (b.xmin, b.xmax) for y in (b.ymin, b.ymax)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return BBox(xmin=min(xs), ymin=min(ys), xmax=max(xs), ymax=max(ys))

    half = struct_size / 2
    for s in out.symbols:
        if s.cls in STRUCTURAL:
            cx, cy = _apply(M, *s.bbox.center)
            s.bbox = BBox(xmin=cx - half, ymin=cy - half, xmax=cx + half, ymax=cy + half)
        else:
            s.bbox = hull(s.bbox)
    out.ignore_regions = [hull(r) for r in out.ignore_regions]
    return out
