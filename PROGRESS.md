# Progress

Spec: [docs/PROPOSAL.md](docs/PROPOSAL.md) **v3** (GraphML primary output, dataset-driven scope, option B).

## Current milestone: 2 (Detection and preprocessing): work done, acceptance needs a decision

Adopted configuration (D-019):
- detector `weights/detector_m2c.pt` (degradation augmentation + repeat-factor sampling, no rot/flip);
- deskew on, crop on, tile overlap 256.

### Milestone 2 acceptance (v3 §6)
- [x] **Degradation simulator deterministic** for a fixed seed: `preprocess/degrade.py`, with tests.
- [x] **Cropping IoU reported** (R-004): 0.950 on Dataset-PID (the ceiling) and 0.959 on OPEN100 train. No false crops on Synthetic, zero GT nodes lost.
- [~] **"Val symbol mAP and node AP beat Milestone 1", with a per-class AP table** (R-007):
  - Clean val symbol mAP@0.5 is **saturated** at 0.996–1.000 for both M1 and M2, so it can only tie.
  - Node AP: Dataset-PID 0.847 → 0.869 (beats); Synthetic 0.927 → 0.925 (tie).
  - On **degraded val (level 2)**, M2 beats M1 by a wide margin: node AP 0.49/0.39 → 0.88/0.90, edge F1 0.50/0.39 → 0.91/0.88.
  - The per-class table is in R-007.

### Done
- **`preprocess/degrade.py`:** 3 levels, skew monotone per sheet, GT transform (D-016).
- **`preprocess/crop.py`:** frame and notes-column detection, enabled by default (D-017).
- **`preprocess/deskew.py`:** projection-profile skew estimate, accurate to 0.014°. It restores level-2 Dataset-PID edge F1 from 0.50 to 0.90.
- **`detect/augment.py`:** degradation augmentation, repeat-factor sampling, rot/flip (now disabled; D-019).
- **Tooling:** `run_baseline.py` gained `--degrade`, `--set`, `--set-detect`, and weights provenance (path + sha1) in every metrics record.
- **GPU use:** 4 detector trainings (m2, m2b, m2c, plus one killed by contention), each under 3 GPU-hours.

### Incident (fixed)
An early robustness run silently loaded 20-iteration smoke-test weights, because `detect.yaml` had changed under it and a killed job's shell loop kept running. Those outputs were deleted. Every metrics file now records the weights hash. Jobs run one at a time, because concurrent training and evaluation slowed both about 20×.

### Open issues for Milestone 3
- **Box precision limits connectivity.** Ankles sit on bbox borders, and an 8 px ankle tolerates about 2.7 px of error. Snapping ankles to the pipe's ink end is a candidate.
- **Dashed lines in Dataset-PID:** non-solid edge AP is about 0.42.
- **OPEN100 line extraction:** oracle edge F1 is 0.62 (diagonal lines, direct symbol–symbol edges, faint lines).
- **Confidence scores** (all structural nodes and edges are 1.0).
- **Real-world rare classes** (pump, tank) can only be judged at the final OPEN100 test. There's no OPEN100 val (D-014).
- **Machine speed:** the GPU looked thermally or power-limited during the latest runs (timings roughly doubled).

## Next concrete step
Pending user confirmation that Milestone 2 is accepted given the saturated clean val. Then start Milestone 3: text mask, dashed lines, OPEN100 line gaps, ankle snapping, confidence calibration.
