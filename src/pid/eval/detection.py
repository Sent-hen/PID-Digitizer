"""Symbol detection metrics via pycocotools.

* `symbol_map` - mAP@0.5 and mAP@0.5:0.95 plus per-class AP@0.5 over symbol classes
  (PID2Graph "Symbols mAP").
* `node_ap` - class-agnostic AP over symbols + crossings + ankles
  (PID2Graph "Nodes AP"; all classes collapsed to one).

pycocotools' default `maxDets=100` is far below the 300-700 nodes per drawing here,
so it is raised explicitly.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pid.io.drawing import Drawing

MAX_DETS = 10_000


def _to_coco(pairs: Sequence[tuple[Drawing, Drawing]], cat_of: dict[str, int],
             ) -> tuple[COCO, COCO | None]:
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    dets: list[dict[str, Any]] = []
    for img_id, (gt, pred) in enumerate(pairs, start=1):
        images.append({"id": img_id, "width": gt.width or 0, "height": gt.height or 0})
        for s in gt.symbols:
            if s.cls not in cat_of:
                continue
            b = s.bbox
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat_of[s.cls],
                         "bbox": [b.xmin, b.ymin, b.width, b.height], "area": b.area,
                         "iscrowd": 0})
        for s in pred.symbols:
            if s.cls not in cat_of:
                continue
            b = s.bbox
            dets.append({"image_id": img_id, "category_id": cat_of[s.cls],
                         "bbox": [b.xmin, b.ymin, b.width, b.height], "score": s.score})
    cats = [{"id": i, "name": n} for n, i in sorted(set(cat_of.items()), key=lambda t: t[1])]
    gt_coco = COCO()
    gt_coco.dataset = {"images": images, "annotations": anns, "categories": cats}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
        dt_coco = gt_coco.loadRes(dets) if dets else None
    return gt_coco, dt_coco


def _evaluate(gt_coco: COCO, dt_coco: COCO | None) -> COCOeval | None:
    if dt_coco is None:
        return None
    ev = COCOeval(gt_coco, dt_coco, iouType="bbox")
    ev.params.maxDets = [MAX_DETS]
    ev.params.areaRng = [ev.params.areaRng[0]]  # "all" only
    ev.params.areaRngLbl = ["all"]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    return ev


def _ap(precision: Any, t_idx: Sequence[int] | slice, k: int | slice) -> float:
    p = precision[t_idx, :, k, 0, 0]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float("nan")


def symbol_map(pairs: Iterable[tuple[Drawing, Drawing]], classes: Sequence[str],
               ) -> dict[str, Any]:
    pairs = list(pairs)
    cat_of = {c: i for i, c in enumerate(classes, start=1)}
    ev = _evaluate(*_to_coco(pairs, cat_of))
    if ev is None:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_AP50": dict.fromkeys(classes, 0.0)}
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    t50 = [int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))]
    per_class = {c: _ap(prec, t50, k) for k, c in enumerate(classes)}
    valid50 = [v for v in per_class.values() if v == v]
    per_class_all = [_ap(prec, slice(None), k) for k in range(len(classes))]
    valid = [v for v in per_class_all if v == v]
    return {
        "mAP50": float(np.mean(valid50)) if valid50 else float("nan"),
        "mAP50_95": float(np.mean(valid)) if valid else float("nan"),
        "per_class_AP50": per_class,
    }


def node_ap(pairs: Iterable[tuple[Drawing, Drawing]], classes: Sequence[str],
            iou: float = 0.5) -> float:
    """Class-agnostic AP at a single IoU over all given classes (collapsed to one)."""
    pairs = list(pairs)
    ev = _evaluate(*_to_coco(pairs, dict.fromkeys(classes, 1)))
    if ev is None:
        return 0.0
    t = [int(np.argmin(np.abs(ev.params.iouThrs - iou)))]
    return _ap(ev.eval["precision"], t, 0)
