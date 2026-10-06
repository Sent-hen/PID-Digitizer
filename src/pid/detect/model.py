"""Detector construction and tiled full-sheet inference (requires the `ml` group)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.models import detection as tvd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pid.detect.merge import merge_detections
from pid.io.drawing import BBox, Symbol
from pid.preprocess.tiling import boxes_to_global, crop_tile, make_tiles, touches_tile_border


def build_model(cfg: Mapping[str, Any], pretrained: bool = True) -> torch.nn.Module:
    m = cfg["model"]
    if m["arch"] != "fasterrcnn_resnet50_fpn_v2":
        raise ValueError(f"unsupported arch {m['arch']}")
    weights = (tvd.FasterRCNN_ResNet50_FPN_V2_Weights[m["pretrained"]]
               if pretrained and m.get("pretrained") else None)
    model = tvd.fasterrcnn_resnet50_fpn_v2(
        weights=weights, min_size=int(m["min_size"]), max_size=int(m["max_size"]),
        box_detections_per_img=int(m["box_detections_per_img"]))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(cfg["classes"]) + 1)
    out: torch.nn.Module = model  # torchvision ships no type stubs
    return out


def to_tensor(gray_tile: NDArray[np.uint8]) -> torch.Tensor:
    t = torch.from_numpy(gray_tile).float().div_(255.0)
    return t.unsqueeze(0).expand(3, -1, -1).contiguous()


class TiledDetector:
    """Callable: working-scale grayscale sheet -> merged symbol detections."""

    def __init__(self, cfg: Mapping[str, Any], weights: str | Path,
                 device: str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(cfg, pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state.get("model", state))
        self.model.to(self.device).eval()
        self.classes = list(cfg["classes"])

    @torch.inference_mode()
    def __call__(self, gray: NDArray[np.uint8]) -> list[Symbol]:
        tc, ic = self.cfg["tiles"], self.cfg["infer"]
        H, W = gray.shape
        tiles = make_tiles(W, H, int(tc["size"]), int(tc["overlap"]))
        all_b, all_s, all_l, all_t = [], [], [], []
        bs = int(ic["batch_size"])
        for i in range(0, len(tiles), bs):
            batch = tiles[i:i + bs]
            imgs = [to_tensor(crop_tile(gray, t)).to(self.device) for t in batch]
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                outs = self.model(imgs)
            for t, o in zip(batch, outs, strict=True):
                keep = o["scores"] >= float(ic["score_threshold"])
                b = boxes_to_global(o["boxes"][keep].float().cpu().numpy().astype(float), t)
                all_b.append(b)
                all_s.append(o["scores"][keep].float().cpu().numpy().astype(float))
                all_l.append(o["labels"][keep].cpu().numpy().astype(np.int64) - 1)
                all_t.append(touches_tile_border(b, t, W, H))
        if not all_b:
            return []
        boxes, scores, labels = merge_detections(
            np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_l),
            np.concatenate(all_t), float(ic["merge_iou"]), float(ic["merge_ios"]))
        boxes = np.clip(boxes, 0, [W, H, W, H])
        return [Symbol(id=f"s{i}", cls=self.classes[int(lab)], score=float(sc),
                       bbox=BBox(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3]))
                for i, (b, sc, lab) in enumerate(zip(boxes, scores, labels, strict=True))]
