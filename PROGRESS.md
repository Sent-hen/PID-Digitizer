# Progress

Spec: [docs/PROPOSAL.md](docs/PROPOSAL.md) **v3** (GraphML primary output, dataset-driven scope, option B).

## Current milestone: 2 (Detection and preprocessing): not started

Milestones 0 and 1 are complete under v3. **RESULTS.md R-003 is the baseline to beat.** R-002 is the superseded first run.

### Milestone 1 (redone for v3, complete 2026-10-05)

Acceptance (v3):
- [x] `make baseline` runs on all test splits with zero failures.
- [x] Every output GraphML passes the official GraphML 1.0 XSD and reloads with the GT loader: 0 invalid across 1 140 files (val + test + oracle).
- [x] Baseline (R-003) and oracle-symbol numbers recorded.

Done in the redo:
- **Option B split:** rare-class-aware pick of 4 OPEN100 training sheets {0, 6, 7, 9}, leaving 8 test sheets (D-014). OPEN100 tiles are oversampled 3×.
- **Detector:** retrained (0.84 GPU-hours); weights in `weights/detector_m1b.pt`.
- **GraphML output:** `export/graphml.py`, with an exact round trip, the vendored XSD, and per-file validation in `run_baseline.py` (D-015). The minimal JSON export is removed.
- **Loader fix:** invalid OPEN100 node ids (e.g. `inlet/outlet3`) are mapped to valid tokens.

Headline test results (R-003), edge F1 / edge mAP:
- dataset_pid: 0.893 / 0.673
- synthetic: 0.925 / 0.907
- open100 (8 sheets): 0.492 / 0.375

On the same 8 OPEN100 sheets, option B raised symbol mAP from 0.397 to 0.704 and edge F1 from 0.395 to 0.492.

### Open issues for later milestones
- **Pump** gets 0 AP on OPEN100, with only 5 training instances (Milestone 2: class imbalance and augmentation).
- **Tank** gets 0.356 AP on OPEN100: large symbols, often bigger than the tile overlap (Milestone 2: tile size/overlap sweep).
- **Dashed lines in Dataset-PID:** non-solid edge AP is 0.42 against 0.92 for solid (Milestone 3).
- **OPEN100 line extraction:** oracle edge F1 is 0.62 (diagonal lines, direct symbol–symbol edges, faint lines) (Milestone 3).
- **Scores:** structural nodes and edges all get score 1.0, so AP collapses to one operating point (Milestone 3 confidence).
- **Ignore regions** aren't used at inference. Frame and title-block false positives are removed only by heuristics (Milestone 2 cropping).

## Next concrete step
Milestone 2 (v3 §6):
1. `preprocess/degrade.py`, deterministic, 3 severity levels.
2. `preprocess/crop.py`, frame and title-block detection evaluated against ignore regions.
3. Detection improvements on val only: class balancing (pump), degradation and rotation augmentations, tile size/overlap sweep.
