"""Build the on-disk tile cache used to train the detector.

Each sheet is rescaled to the pipeline's working scale, cut into overlapping tiles,
and every tile is saved as a grayscale PNG with its (clipped) symbol boxes in an
index JSON. Structural nodes (crossing/ankle) are not detector targets.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pid.io.pid2graph import load_drawing, ref_from_uid
from pid.pipeline import STRUCTURAL, load_gray
from pid.preprocess.tiling import (
    crop_tile,
    make_tiles,
    rescale_factor,
    rescale_image,
    scale_drawing,
)


def tile_sheet(uid: str, out_dir: Path, det_cfg: Mapping[str, Any],
               target_long_side: int | None, seed: int) -> list[dict[str, Any]]:
    tcfg = det_cfg["tiles"]
    classes = list(det_cfg["classes"])
    ref = ref_from_uid(uid)
    gt = load_drawing(ref, with_size=False)
    gray = load_gray(ref.image)
    f = rescale_factor(gray.shape[1], gray.shape[0], target_long_side)
    work = rescale_image(gray, f)
    gt = scale_drawing(gt, f)
    H, W = work.shape
    boxes = np.array([s.bbox.as_xyxy() for s in gt.symbols if s.cls not in STRUCTURAL],
                     float).reshape(-1, 4)
    labels = [classes.index(s.cls) for s in gt.symbols if s.cls not in STRUCTURAL]
    rng = random.Random(f"{seed}:{uid}")
    records = []
    for i, t in enumerate(make_tiles(W, H, int(tcfg["size"]), int(tcfg["overlap"]))):
        cb = np.clip(boxes, [t.x0, t.y0, t.x0, t.y0], [t.x1, t.y1, t.x1, t.y1])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        vis = (cb[:, 2] - cb[:, 0]).clip(0) * (cb[:, 3] - cb[:, 1]).clip(0)
        keep = vis >= float(tcfg["min_visible"]) * np.maximum(area, 1e-9)
        if not keep.any() and rng.random() >= float(tcfg["empty_keep_prob"]):
            continue
        name = f"{uid.replace('/', '_')}_{i:03d}.png"
        cv2.imwrite(str(out_dir / name), crop_tile(work, t))
        local = cb[keep] - np.array([t.x0, t.y0, t.x0, t.y0], float)
        records.append({"file": name, "uid": uid,
                        "boxes": local.round(2).tolist(),
                        "labels": [labels[j] for j in np.nonzero(keep)[0]]})
    return records


def build_cache(split_names: list[str], split_dir: Path, out_dir: Path,
                det_cfg: Mapping[str, Any], target_long_side: int | None,
                workers: int = 8) -> Path:
    """Tile every sheet in the given split lists; returns the index path."""
    from concurrent.futures import ProcessPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.json"
    uids = [u for s in split_names for u in (split_dir / f"{s}.txt").read_text().split()]
    seed = int(det_cfg["seed"])
    recs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(tile_sheet, u, out_dir, det_cfg, target_long_side, seed) for u in uids]
        repeat = {k: int(v) for k, v in det_cfg["tiles"].get("oversample", {}).items()}
        for n, (u, fut) in enumerate(zip(uids, futs, strict=True), 1):
            # Oversampling repeats index records (not files): a subset with factor k
            # contributes each of its tiles k times per epoch.
            recs.extend(fut.result() * repeat.get(u.split("/")[0], 1))
            if n % 50 == 0:
                print(f"tiled {n}/{len(uids)} sheets, {len(recs)} tiles")
    index.write_text(json.dumps({"splits": split_names, "tiles": recs}))
    return index
