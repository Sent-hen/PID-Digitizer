# Progress

## Current milestone: 0 (Dataset audit, scaffolding, evaluation harness): complete, awaiting review

Per the proposal (§8), pipeline code (Milestone 1) does not start until the plan adjustments below are reviewed.

### Done (Milestone 0)
- [x] Dataset audit: `scripts/inspect_dataset.py` writes `docs/dataset_stats.{md,json}` and overlays in `docs/dataset_samples/` (6 full sheets plus 6 native-resolution crops covering all subsets).
- [x] `DATASET_NOTES.md`: format, license (CC BY-SA 4.0), attribution, quirks, official protocol, and published baselines.
- [x] Canonical pydantic model `src/pid/io/drawing.py` (`Drawing`, `Symbol`, `Edge`, `BBox`, `Text`, `Port`) and loader `src/pid/io/pid2graph.py`.
- [x] Round trip (dataset → `Drawing` → overlay) checked visually on 6 sampled drawings across all 3 subsets. Boxes and edges align with the drawings.
- [x] Splits: `scripts/make_splits.py` with near-duplicate grouping by ink-map Jaccard. Lists are in `splits/`.
- [x] Harness:
  - `eval/match.py`: IoU, gIoU, greedy and Hungarian matching.
  - `eval/detection.py`: pycocotools symbol mAP and node AP.
  - `eval/connectivity.py`: edge P/R/F1, line-type accuracy, and the official PID2Graph edge mAP.
  - `eval/report.py`, plus `scripts/evaluate.py`.
- [x] Oracle tests: GT scores 1.0 on every metric (unit fixtures and all three test splits). Known perturbations give exact expected drops (`tests/test_oracle.py`).
- [x] Scaffolding:
  - uv and `uv.lock`, plus ruff, mypy (strict) and pytest.
  - `configs/data.yaml`, the Makefile, and GitHub Actions CI (CPU-only, fixtures only).
  - Hardware recorded in DECISIONS.md D-001.

Checks: `ruff` clean, `mypy --strict` clean, 27 unit tests pass, 3 dataset tests pass (`pytest -m dataset`).

### Open issues
- `make` isn't installed on the dev machine (Windows). The Makefile serves CI and WSL; README lists the equivalent `python` commands.

## Proposed plan adjustments (need your decision before Milestone 1)

1. **There is no released training data. All three subsets are the paper's test sets.**
   - We train on our Dataset-PID and Synthetic train splits and report OPEN100 as a held-out real-world test.
   - Numbers are reported under the official protocol but are **not directly comparable** to paper Table III (different training data, smaller test sets). README and RESULTS will say so explicitly.
2. **inlet/outlet exists only in OPEN100, and pump/tank only in Synthetic and OPEN100.**
   - **Option A (default):** keep OPEN100 fully held out and report inlet/outlet as untrained (AP 0, footnoted).
   - **Option B:** move 4 of the 12 OPEN100 sheets into training (8 remain for test). That gives a little real-world data but a noisier test set.
3. **Drop the "junction vs crossing" metric** (Milestone 3) as a ground-truth metric.
   - The GT links every line intersection into a connected `crossing` node, and only 2 degree-4 crossings exist in OPEN100, so there are no labels for it.
   - Instead, the graph assembly will follow the dataset convention so that it scores correctly. Explicit junction/crossing logic stays as an optional mode, unit-tested on synthetic fixtures only.
   - Separating junctions from pass-through crossings for real-world use is listed under Limitations.
4. **Ports:** the dataset has no port annotations, but `ankle` (raw `connector`) nodes sit exactly where pipes meet symbols.
   - Use ankles as port supervision.
   - Structural nodes (crossing/ankle) count in Nodes AP and edge mAP, so the pipeline must produce them. Plan: derive them from line geometry (bends, intersections, symbol-boundary contacts), with "detect them as 8×8 classes" as an ablation.
5. **Scale normalization:** median symbol size differs about 5× between subsets (valve: 66 px, 45 px, 20 px). Add a config option to rescale each sheet to a target median symbol size, or to a fixed long side as the paper does (7000 px), before tiling.
6. **OCR (Milestone 3):**
   - Synthetic text is random strings, and no subset has text GT.
   - Hand-label about 300 text crops from Dataset-PID (realistic tags such as `AB-04958`, `CV-68416`) and OPEN100.
   - Develop the tag grammar on Dataset-PID and OPEN100 only.
7. **Line-type accuracy** is reported on Dataset-PID and Synthetic only (OPEN100 has no non-solid lines).
8. **Cross-subset test:** "train synthetic → test real-derived" is weak here because class sets barely overlap between subsets. OPEN100 serves as the cross-domain test.

## Next concrete step
After the adjustments are approved, start Milestone 1:
- `preprocess/tiling.py` with coordinate mapping and a round-trip test.
- Choose the detector (torchvision Faster R-CNN vs Apache-2.0 RT-DETR; record the license in DECISIONS).
- Build the tile dataset from `splits/*_train.txt`.
