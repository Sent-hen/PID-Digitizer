# Decisions

Each entry records the choice, the reason, and the alternative that was rejected.

## D-001 Hardware (recorded 2026-10-02)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GB VRAM, driver 610.88, compute capability 8.9 |
| CPU | Intel Core Ultra 9 185H, 22 logical cores |
| RAM | 15.4 GB |
| Disk | ~650 GB free on C: |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.13 (project venv; 3.14 also installed but has thinner wheel coverage for torch/pycocotools) |

Implications:
- **8 GB of VRAM** means small or medium detectors, mixed precision, and roughly 1024² tiles at batch 4–8. Full 7168×4561 sheets never go to the GPU whole.
- **15 GB of RAM** means images can't all be cached in memory: one decoded sheet is about 98 MB as RGB, or 33 MB as grayscale. Tiles go to a disk cache.
- **Tile cache estimate:** 1024² tiles at 25% overlap give about 60 tiles per Dataset-PID sheet. At about 100 KB per PNG tile, the 524 training sheets come to roughly 3 GB, plus Synthetic at about 1.5 GB. That's negligible against the free disk.

## D-002 Environment: uv inside a project venv

`uv` was not installed system-wide, so `py -3.13 -m venv .venv` was created and `uv` installed into it. Dependencies are pinned in `uv.lock`. We rejected installing uv globally to avoid changing the user's system Python.

## D-003 Splits: our own, per drawing, OPEN100 fully held out

- PID2Graph ships no training data. All three subsets are the paper's test sets (DATASET_NOTES.md).
- **Splits** (seed 1234, `configs/data.yaml`):
  - Dataset-PID: 350 / 75 / 75 (train / val / test).
  - Synthetic: 174 / 38 / 38.
  - OPEN100: all 12 drawings are test only.
- **Why OPEN100 is held out:** it is the only real-world subset and has only 12 sheets. Training on any of it would leave too little for a credible test. As a cross-domain test (train on generated data, test on real), it is the most honest number we can report.
- **Rejected:** a 5-fold CV over OPEN100 (8–9 train / 3–4 test per fold). It's too few to train on, and it would mix tuning and testing on the same 12 sheets.
- **Consequence:** inlet/outlet appears only in OPEN100, so no training data exists for that class. It will score 0 AP unless we add data. This is listed as a proposed plan adjustment in PROGRESS.md.

## D-004 Near-duplicate grouping: ink-map Jaccard, not pHash

- **pHash was tried first and rejected.** On sparse line art on a white background, the low-frequency DCT is nearly constant. Clearly unrelated Synthetic sheets (verified visually: synthetic/17 vs synthetic/66) had Hamming distance 0, and union-find chained 103 Synthetic drawings into one group.
- **Replacement:** a 96×64 binary ink map, where a cell is 1 if it holds any pixel darker than 128. Ignore regions (frame and title block, identical on every Dataset-PID sheet) are masked out, then pairwise Jaccard is computed. Exact-duplicate annotation graphs are grouped too.
- **Result:** the nearest-neighbour Jaccard maximum is 0.44, well below the 0.6 threshold, so every group is a singleton and the split is effectively per drawing. If a future subset has near-duplicates, the same script groups them.

## D-005 Evaluation implementation choices

- **Symbol and node mAP:** pycocotools (BSD-2), not hand-rolled. pycocotools' default `maxDets=100` silently truncates detections, and these sheets have 300–700 nodes, so we set `maxDets=[10000]` and use the "all" area range only.
- **Edge mAP:** no library exists. We use a small all-point AP function (`eval/ap.py`) covered by unit tests with hand-computed values.
- **Two connectivity metrics:**
  - *Primary:* **edge P/R/F1** with greedy IoU ≥ 0.5 node matching. It is interpretable and allows per-drawing distributions.
  - *Official:* **PID2Graph edge mAP** (Hungarian on gIoU, no threshold), for comparability.
  - We keep both because Hungarian matching without a threshold can pair a prediction with a far-away GT node. The official metric is reported as specified, and the P/R/F1 metric guards against that.
- **Line type** is excluded from edge P/R/F1 and reported separately as line-type accuracy over TP edges. The official edge mAP includes it, per the paper.

## D-006 Library licenses (so far)

| library | license | use |
|---|---|---|
| numpy, scipy, networkx | BSD-3 | core |
| pillow | MIT-CMU (HPND) | images |
| pydantic | MIT | data model |
| PyYAML | MIT | configs |
| opencv-python-headless | Apache-2.0 (OpenCV) / MIT (wheels) | image ops (Milestone 1+) |
| matplotlib | PSF-based (matplotlib license) | figures |
| pycocotools | BSD-2 | mAP |
| pytest, ruff, mypy | MIT | dev |

No AGPL dependency. **Ultralytics YOLO (AGPL-3.0) is not used.** Detector candidates for Milestone 1 are torchvision Faster R-CNN / RetinaNet / FCOS (BSD-3) or an Apache-2.0 RT-DETR. To be decided at Milestone 1 and recorded here.
