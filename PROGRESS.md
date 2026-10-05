# Progress

## Current milestone: 2 (Ingestion paths and detection improvements): not started

Milestones 0 and 1 are complete. Milestone 1's numbers (RESULTS.md R-002) are the baseline every later milestone must beat.

### Milestone 1: walking skeleton (complete, 2026-10-04)

Acceptance:
- [x] `make baseline` runs end to end on all test splits with zero failures. Failures are caught, logged and counted in `metrics_<split>.json`.
- [x] Baseline detection and connectivity numbers are recorded (RESULTS R-002), along with the oracle-symbol diagnostic (R-001).

Done:
- **Tiling:** `preprocess/tiling.py`, with overlapping tiles, coordinate mapping and rescaling to a 7000 px long side. Round-trip tests pass within 1 px.
- **Detector:** torchvision Faster R-CNN R50-FPN v2, BSD-3 (D-008). Trained 0.87 GPU-hours on 19 398 train tiles; weights are in `weights/detector_m1.pt` (gitignored, 166 MB).
- **Tile merging:** `detect/merge.py`, complete-first NMS using max(IoU, IoS) and unioning truncated boxes (D-009). Unit-tested at tile seams.
- **Lines and graph:** `lines/extract.py` and `graph/assemble.py`, using orthogonal morphology and dataset-convention graph assembly (D-010). Synthetic fixtures cover the inline valve, T, X, bend, dashed line, and frame plus text.
- **Export and runner:** `export/json_export.py` (minimal JSON) and `scripts/run_baseline.py`.
- **CUDA:** torch 2.14.1+cu132, matched to the installed CUDA 13.4 toolkit and the driver's 13.3 ceiling (D-012).

Headline test results (R-002), edge F1 / edge mAP:
- dataset_pid: 0.903 / 0.684
- synthetic: 0.932 / 0.908
- open100: 0.409 / 0.369

### Milestone 0 (complete)
Covers the dataset audit, the canonical model, splits, the evaluation harness with oracle tests, and scaffolding. See git history and DATASET_NOTES/DECISIONS.

### Open issues and observations for later milestones
- **Detection on real drawings is the bottleneck.** OPEN100 symbol mAP is 0.416. Pump and inlet/outlet have no usable training data, and tank only has Synthetic-style examples. Milestone 2 should try class imbalance handling and degradation augmentations. The decision to keep OPEN100 held out (D-007) caps what's achievable here.
- **Dashed lines in Dataset-PID** have non-solid edge AP of 0.43 against 0.93 for solid (Milestone 3).
- **OPEN100 line extraction** reaches oracle edge F1 of only 0.65: diagonal lines, direct symbol–symbol edges and faint signal lines (Milestone 3).
- **Scores:** all structural nodes and edges get score 1.0, so AP collapses to a single operating point. Confidence scoring comes in Milestone 3.
- **Ignore regions** (title block) aren't used at inference. The frame and title block are suppressed only by heuristics (D-010). Border cropping is part of Milestone 2.

## Next concrete step
Milestone 2:
1. `io/classify.py`, PDF vector vs raster classification, with ≥4 test PDFs.
2. `preprocess/degrade.py`, a deterministic degradation simulator.
3. Detection improvements, measured on val only: degradation and rotation augmentations, class balancing, tile size/overlap sweeps.
