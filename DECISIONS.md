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

## D-003 Splits: our own, per drawing, OPEN100 fully held out (OPEN100 part superseded by D-014)

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

## D-007 Plan adjustments adopted at the start of Milestone 1 (2026-10-02)

The user approved moving to Milestone 1 without choosing among the proposed adjustments (PROGRESS.md), so the defaults were taken:
- **OPEN100 stays fully held out.** inlet/outlet has no training data and will score 0 AP; this is footnoted in results.
- **No GT junction-vs-crossing metric.** Assembly follows the dataset convention: any H/V meeting is one `crossing` node.
- **Structural nodes come from line geometry**, not the detector. `ankle` nodes are placed where a line meets a symbol's bbox border. Every Dataset-PID/Synthetic GT ankle lies exactly on its symbol's border (measured over 40 sheets each), and every structural–structural GT edge there is axis-parallel.
- **Every sheet is rescaled to a 7000 px long side** before detection and line extraction, as the paper does (paper: 4500×7000). This upscales OPEN100 about 2.7× so its symbols match the training scale.

## D-008 Detector: torchvision Faster R-CNN R50-FPN v2 (BSD-3)

- **Model:** `fasterrcnn_resnet50_fpn_v2`. The code is torchvision, BSD-3. The weights are COCO_V1 pretrained, distributed by torchvision under BSD-3, and trained on COCO, whose annotations are CC BY 4.0. The paper's modular baseline also uses Faster R-CNN, which helps comparability.
- **Rejected:**
  - Ultralytics YOLO: AGPL-3.0.
  - RT-DETR via Ultralytics: AGPL-3.0.
  - RT-DETR via HF transformers (Apache-2.0): a viable later option, but it needs more VRAM and longer schedules than an 8 GB laptop GPU allows for a short baseline.
  - MMDetection (Apache-2.0): a heavy dependency stack.
- **Setup:**
  - **Tiles:** 1024² with 256 overlap, fed at native size (`min_size = max_size = 1024`).
  - **Targets:** GT boxes cut by a tile are kept if at least 50% is visible. 10% of symbol-free tiles are kept as negatives.
  - **Training:** 7 symbol classes, AMP, SGD with cosine schedule, 6000 iterations at batch 4 (about 1.2 epochs over 19 398 tiles).
- **Tile cache:** `.cache/tiles/train`, 1.2 GB, built in 33 s with 8 processes.
- **Torch packaging:**
  - torch/torchvision come from the PyTorch cu132 index (see D-012) in uv group `ml`, which is a default group locally so `uv sync` won't uninstall it.
  - CI runs `uv sync --no-group ml` and `uv run --no-sync`, and the merge/tiling code is numpy-only, so CI never needs torch.

## D-009 Tile merging: complete-first NMS with intersection-over-smaller

- Plain IoU NMS fails at tile seams. A symbol cut by a tile edge yields a fragment whose IoU with the full box from the neighbouring tile is below 0.5, so both survive.
- **What `detect/merge.py` does:**
  - Boxes touching an *interior* tile edge are marked truncated and rank after complete boxes.
  - Duplicates are found with max(IoU, IoS), where IoS is the intersection divided by the smaller box's area.
  - Two truncated boxes that are duplicates merge into their union (for objects larger than the overlap).
- Unit-tested in `tests/test_tiling_merge.py`.
- **Rejected for now:** weighted box fusion, which the proposal lists as an optional Milestone 2 comparison.

## D-010 Lines: orthogonal morphology instead of skeletonization

The proposal suggested skeletonize → segments. We chose directional morphology instead: close with a short kernel, open with a long horizontal or vertical kernel, and take each component's centroid as the centreline.
- **Why:**
  - The GT is axis-aligned (D-007), and an 8×8 node must land within about 2.7 px of the GT point to reach IoU 0.5. Component centroids give sub-pixel centrelines.
  - Skeleton junctions wander at corners and thick lines.
- **Filters**, each motivated by a failure seen on *train* sheets:
  - runs longer than 80% of the sheet are the frame;
  - candidates with ink alongside them on more than 50% of their length are merged text;
  - crossing nodes used by only one segment and lying strictly inside it are text or symbol spurs, since the GT never has a degree-2 node on a straight run;
  - connected components with no symbol are dropped (title-block tables, text).
- **Known gaps (Milestone 3):** diagonal lines (6% of OPEN100 structural edges), direct symbol–symbol edges (OPEN100 only), and light-gray thin signal lines in OPEN100.
- **Disclosure:** the text-context filter was partly motivated by inspecting OPEN100 sheets 0 and 1, which are test drawings. No thresholds were fitted numerically on OPEN100. All later tuning uses train/val only.

## D-011 AP with tied scores

The pipeline assigns score 1.0 to all structural nodes and edges for now. The first edge-AP implementation broke ties by list order, which made AP depend on output order. `eval/ap.py` now treats each tied-score group as one threshold, as sklearn does, with a regression test.

## D-012 CUDA build of torch: cu132 (2026-10-04)

- **Machine:**
  - CUDA Toolkit 13.4 installed (nvcc V13.4.92).
  - Driver 610.88 reports CUDA UMD 13.3, the highest runtime it supports.
- **Choice:** the PyTorch cu132 build (torch 2.14.1+cu132, torchvision 0.29.1+cu132), replacing the earlier cu128 pin.
- **Why cu132:**
  - PyTorch publishes no 13.4 Windows wheels. The cu134 index holds only old aarch64 placeholders, and cu133 doesn't exist.
  - cu132 is the newest CUDA 13 build at or below the driver's 13.3.
  - It shares major version 13 with the installed toolkit. torch's `cpp_extension` checks only the major version, so building custom CUDA extensions with nvcc 13.4 works.
  - PyTorch wheels bundle their own CUDA runtime, so the toolkit's version isn't what torch loads at runtime.
- **Rejected:**
  - Staying on cu128: a CUDA 12 runtime alongside a CUDA 13 toolkit, so extension builds would mismatch.
  - cu130: older, no benefit.

## D-013 Proposal v3 scope (2026-10-04, user-approved)

The user approved:
- GraphML as the primary output, with DEXPI-structured XML as secondary;
- option B (4 OPEN100 sheets in training, chosen rare-class-aware, oversampled about 3×);
- four scope cuts: vector PDF → stretch; OCR → text mask only; drop junction/ports/off-page; merge JSON into GraphML;
- removing items that are only for show.

The full change table is in `docs/PROPOSAL.md` §9.

- **Why GraphML over DEXPI as primary:**
  - It's the dataset's own GT format, so predictions and GT share one loader and round trip exactly.
  - Our classes (7 coarse classes plus structural nodes) map only partly to DEXPI component classes.
  - DEXPI stays as the interoperability export.
- **Rejected:**
  - DEXPI as primary: the conversion is lossy, and evaluation would need a DEXPI-to-graph converter.
  - AutomationML/PandIX: less P&ID tool support and heavier to generate and validate.
- **Milestone 1 is redone** under v3: new OPEN100 split, retrained detector, GraphML export.

## D-014 Option B split and oversampling (2026-10-04, user-approved)

- **OPEN100:** 4 sheets go to training, and 8 stay in test with no val split.
- **How the 4 are chosen** (`io/splits.rare_class_aware_pick`, seed 1234), out of all C(12,4) = 495 combinations:
  - every class keeps at least 1 instance in test;
  - every rare class (tank, pump, inlet/outlet) has at least 1 instance in train;
  - among those, minimize the squared deviation of each rare class's train share from 4/12.
- **Result:**
  - train = {0, 6, 7, 9}: tank 12/35, pump 5/16, inlet/outlet 29/96;
  - test = {1, 2, 3, 4, 5, 8, 10, 11}, which still contains pumps (sheets 1, 3, 5).
- **Other subsets:** the split uses its own RNG, so the Dataset-PID and Synthetic lists are byte-identical to Milestone 0's.
- **Oversampling:** OPEN100 train tiles are repeated 3× in the tile index (`detect.yaml: tiles.oversample`): 153 unique tiles become 459 of 19 857. Without it, real sheets make up 0.8% of tiles. The paper likewise augmented real P&IDs 3× more than synthetic ones.
- **Rejected:**
  - a random seeded pick, which could put all pump sheets on one side;
  - an OPEN100 val split, which with 12 sheets would leave too few for train or test.
- **Consequence:** there's no OPEN100-specific validation. Tuning uses Dataset-PID/Synthetic val, and the 8-sheet OPEN100 test is run once per final config.

## D-015 GraphML export implementation

- **Attribute names:** the GT names and raw label spellings, so `read_graphml` reuses the GT loader.
- **Floats:** written with `repr`, so Drawing → GraphML → Drawing is exact and GraphML → Drawing → GraphML is byte-identical (tested).
- **Edge order and orientation:** networkx scrambles both for undirected graphs, so they're restored from the file.
- **Validation:**
  - against the official GraphML 1.0 XSD, vendored in `src/pid/export/xsd/` with local `schemaLocation`s for offline CI, using `lxml` (BSD-3, new light dependency);
  - `run_baseline.py` validates and reloads every output, counting `invalid_graphml`.
- **Removed:** the minimal JSON export. JSON remains only the API transport, per proposal v3.

## D-016 Degradation simulator design (Milestone 2)

- **Operations**, each level applies all of them in order: contrast loss → blur → skew → noise and speckle → resolution loss → JPEG. Level 3 adds fax-like binarization. Parameters are in `configs/degrade.yaml`.
- **Determinism:** all randomness comes from `default_rng([seed, crc32(uid), level])`.
- **Skew** direction and relative magnitude (0.5–1× the level maximum) are fixed *per sheet* and scaled by level. Without this, a sheet could be skewed less at level 3 than at level 2, which would make robustness curves non-monotone.
- **Ground truth** is mapped through the skew: symbols get the axis-aligned hull of their rotated corners, and structural nodes are re-centred at a fixed 8 px.
- **Rejected:** per-operation severity sweeps, which would cost 6+ evaluation runs per model with no extra decision value.

## D-017 Frame/title-block cropping from ruling lines

- **What the GT content box is:** for Dataset-PID it's a fixed rectangle, `[400, 304, 5568, 4304|4312]` on every sheet, set slightly inside the drawn frame and left of the notes column. A line-based detector therefore tops out at IoU about 0.95.
- **Method:**
  1. Downscale by block-minimum (area averaging erased thin 1–2 px frames, e.g. sheet 3).
  2. The frame is the outermost long horizontal and vertical rules. It must hug at least 3 sheet edges (OPEN100 puts the title block outside the frame), and its top and bottom rules must cover at least 90% of its width.
  3. A notes or title-block column is a long interior vertical rule in the right 40%.
  4. Content outside the box is blanked before line extraction, and detections centred outside it are dropped.
- **Train/val result:**
  - IoU 0.950 on Dataset-PID and 0.959 on OPEN100 train;
  - no crops on any Synthetic sheet (an earlier version cropped 3, losing 94 GT nodes; fixed by the edge and coverage checks);
  - zero GT nodes outside the box.
- **Rejected:**
  - using the subset identity to apply the fixed rectangle: that's leakage, since the subset is unknown at inference;
  - a learned crop model: no need, given IoU at the ceiling.

## D-018 Detector augmentation and rare-class sampling

- **90° rotations and horizontal flips** are treated as label-preserving because none of the 7 classes encodes direction. `arrow` covers all orientations, and valves appear in both orientations on real sheets. The proposal asked to note this for directional symbols; for this dataset the label set makes it safe.
- **Degradation augmentation:** 50% of tiles get a random level-1 to level-3 degradation, with boxes mapped through the skew. This targets robustness on degraded inputs, which is measurable on degraded val.
- **Repeat-factor sampling** (threshold 0.04) upweights tiles with tank (4.9% of tiles), pump (2.9%) and inlet/outlet (0.5%), at most 2.9×.
- **Limitation:** pump tiles are almost all Synthetic-style pumps, and OPEN100 has no val split (D-014). So the effect on real pumps can only be seen at the final OPEN100 test, not on val.

## D-019 Adopted Milestone 2 configuration (2026-10-05)

- **Detector:** m2c. Degradation augmentation (p = 0.3, no skew) plus repeat-factor sampling, **without rot90/flip**.
  - Rot/flip were rejected after the m2 → m2b → m2c ablation (R-005 to R-007). They cost about 0.08 mAP@0.5:0.95 and 0.12 edge F1 on clean Dataset-PID, because that subset's boxes follow an asymmetric convention (larger left/top margins) that a rotation-invariant model can't learn.
  - This supersedes the rot/flip part of D-018.
- **Pipeline:** deskew on, crop on, inference overlap 256 (512 gave no gain at 2× cost).
- **Rejected:** m1b, the M1 detector. It's 0.003–0.010 better in clean edge F1, but much worse on degraded input (Synthetic level 2: symbol mAP 0.859 vs 0.996, edge F1 0.747 vs 0.877).

## D-020 Milestone 2 sign-off (2026-10-05)

- **Decision:** Milestone 2 is accepted with clean-val symbol mAP@0.5 saturated at 0.996–1.000 for both M1 and M2, so that metric can only tie.
- **How "beats M1" was judged instead:**
  - node AP: Dataset-PID 0.847 → 0.869;
  - degraded val (level 2), where M2 wins by a wide margin (R-007).
- **Rejected:**
  - rewriting the acceptance metric before continuing;
  - spending a one-time OPEN100 test run on an intermediate configuration.

## D-021 Milestone 3 line/graph decisions (2026-10-05)

- **Ankles as ports:** two graph rules from the GT invariant that ankles have degree 2:
  - drop edges between two ankles of the same symbol;
  - iteratively drop ankles left with only their symbol edge.
  - These remove ports created by text strokes and by symbol outlines drawn just outside the box (diagnosed on val sheet 147). Oracle val, Dataset-PID: edge F1 0.938 → 0.954.
- **Dashed joins:** non-solid segments get `dashed_extra_tol` = 20 px on join and snap. A dashed run ends at its last dash, up to one gap (about 16 px) short of the line or symbol it meets (diagnosed on val sheet 333). Oracle non-solid recall 0.778 → 0.898.
- **Ink threshold 150 → 180:**
  - better on OPEN100 train (edge F1 0.689 → 0.719) and Dataset-PID val (0.953 → 0.955), Synthetic unchanged;
  - degraded paper stays above 180 (level-2 paper 220, noise σ 12; level 3 is binarized).
- **Mask padding:** 1 px kept; 3 and 5 px made no difference with detector boxes.
- **Text mask:** substituted, not implemented as a separate model.
  - Proposal v3 asked for "text detection only, permissive detector". Text-induced false structure is already handled by the side-ink filter (D-010), the ankle-port rule above, and symbol-free component pruning. A "text next to a line" fixture passes.
  - A learned text detector (CRAFT/EasyOCR/PaddleOCR) would be a heavy new dependency that needs user approval (CLAUDE.md), and the remaining errors on val aren't text-dominated.
  - Revisit if the error analysis (Milestone 5) shows text-induced errors.
- **Diagonal lines are not supported** (orthogonal-only line stage). They're 6% of OPEN100 structural edges and absent from the generated subsets. Fixture `test_diagonal_line` is a strict xfail, and this goes under Limitations.
- **Remaining OPEN100 misses** (inspected on train sheets 6 and 9):
  - very short nozzle stubs between small symbols and tank walls, below `min_length`;
  - symbols nested inside a large symbol's box, blanked along with the outer box.
- **Confidence:** a logistic model over 11 geometric/detector features (graph/confidence.py), fitted on val.
  - Two-fold CV by drawing gives ECE and edge mAP on held-out drawings.
  - The mid-range (0.4–0.8) is overconfident, but those bins hold fewer than 3% of edges. A second calibration step (isotonic) was rejected for so little mass.
  - Structural node confidence = mean of incident edge confidences.
