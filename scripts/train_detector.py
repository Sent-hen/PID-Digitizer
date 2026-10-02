"""Fine-tune the symbol detector on cached tiles.

Usage:
  python scripts/train_detector.py [--config detect] [--build-cache] [--iterations N]
Writes weights to `train.out` and a JSONL training log to outputs/train_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from pid.config import load_config, resolve
from pid.detect.model import build_model, to_tensor
from pid.detect.tiles import build_cache


class TileDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, index: Path) -> None:
        self.root = index.parent
        self.recs: list[dict[str, Any]] = json.loads(index.read_text())["tiles"]

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r = self.recs[i]
        img = cv2.imread(str(self.root / r["file"]), cv2.IMREAD_GRAYSCALE)
        boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(r["labels"], dtype=torch.int64) + 1  # 0 = background
        return to_tensor(img), {"boxes": boxes, "labels": labels}


def collate(batch: list[Any]) -> tuple[list[Any], list[Any]]:
    imgs, targets = zip(*batch, strict=True)
    return list(imgs), list(targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="detect")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--workers", type=int, default=8, help="processes for tile caching")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pcfg = load_config("pipeline")
    dcfg = load_config("data")
    tcfg, tr = cfg["tiles"], cfg["train"]
    cache = resolve(tcfg["cache_dir"]) / "train"
    index = cache / "index.json"
    if args.build_cache or not index.exists():
        t0 = time.time()
        build_cache(tcfg["train_splits"], resolve(dcfg["splits"]["dir"]), cache, cfg,
                    pcfg["preprocess"].get("target_long_side"), args.workers)
        print(f"tile cache built in {time.time() - t0:.0f}s")

    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds DataLoader worker numpy RNGs via torch
    ds = TileDataset(index)
    print(f"{len(ds)} training tiles")
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=int(tr["batch_size"]), shuffle=True, generator=g,
                    num_workers=int(tr["num_workers"]), collate_fn=collate, drop_last=True,
                    persistent_workers=int(tr["num_workers"]) > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(tr["lr"]), momentum=float(tr["momentum"]),
                          weight_decay=float(tr["weight_decay"]))
    iters = int(args.iterations or tr["iterations"])
    warm = int(tr["warmup_iters"])

    def lr_at(it: int) -> float:
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, iters - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=bool(tr["amp"]) and device.type == "cuda")
    log = resolve("outputs/train_log.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=resolve(".")).stdout.strip()
    run_id = time.strftime("%Y%m%d-%H%M%S")

    it, t0 = 0, time.time()
    while it < iters:
        for imgs, targets in dl:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % int(tr["log_every"]) == 0 or it == iters:
                rec = {"run": run_id, "commit": commit, "iter": it,
                       "lr": sched.get_last_lr()[0], "loss": float(loss),
                       **{k: float(v) for k, v in losses.items()},
                       "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec))
                with log.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            if it >= iters:
                break

    out = resolve(tr["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "run": run_id, "commit": commit,
                "iterations": iters}, out)
    print(f"saved {out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
