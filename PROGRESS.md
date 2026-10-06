# Progress

Spec: [docs/PROPOSAL.md](docs/PROPOSAL.md) **v3** (GraphML primary output, dataset-driven scope, option B).

## Current milestone: 4 (Export: GraphML schema doc + DEXPI-structured XML): not started

Milestones 0–3 are complete. The current configuration is the M3 pipeline (RESULTS R-008):
- detector m2c;
- deskew, crop, and the ankle-port rule;
- dashed-join tolerance and the paper-relative ink threshold;
- calibrated confidence.

### Milestone 3 (complete 2026-10-05; D-021, D-022)
- [x] **Val edge F1 / edge mAP vs M1** (R-008):
  - Dataset-PID: 0.886 → 0.917 F1, 0.647 → 0.909 mAP.
  - Synthetic: 0.904 → 0.918 mAP, but F1 0.915 → 0.905 (−0.010, from the M2 detector trade-off; line stage unchanged per oracle). Accepted in D-022.
- [x] **Fixtures pass:** T, X, bend, inline valve, dashed line, outline-outside-box, text-stroke port, text next to a line. Diagonal line is a strict xfail (limitation).
- [x] **Reliability diagram:** `docs/reliability_diagram.png`. 2-fold CV ECE is 0.015, and out-of-sample edge mAP goes from 0.787 to 0.905.
- [x] **Confidence-colored visualization:** `render_overlay(color_by="confidence")`, sample in `docs/`.
- **What changed:**
  - ankle-port rule (Dataset-PID oracle F1 0.938 → 0.954);
  - dashed joins (non-solid recall 0.78 → 0.90);
  - threshold 180 with paper cap;
  - confidence model;
  - `scripts/analyze_errors.py`.
- **Substitutions and limitations:**
  - text handled by heuristics instead of a learned text detector, which would be a heavy dependency (D-021);
  - diagonal lines and OPEN100 nested-symbol / short-stub connections aren't handled.

### Milestone 2 acceptance (v3 §6, accepted in D-020)
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

### Open issues (carried forward)
- **Box precision limits connectivity.** Ankles sit on bbox borders, and an 8 px ankle tolerates about 2.7 px of error.
- **Crossing positions:** Dataset-PID GT marks crossings at the stroke's top-left edge, Synthetic at its centre. That gives up to 2.5 px offsets on thick Dataset-PID strokes. It's a per-subset convention, so it isn't corrected (no subset identity at inference).
- **OPEN100 line stage:** oracle edge F1 is 0.72 on train sheets. The gaps are diagonal lines, short nozzle stubs, nested symbols, and direct symbol–symbol edges.
- **Real-world rare classes** (pump, tank) can only be judged at the final OPEN100 test, since there's no OPEN100 val (D-014).
- **Confidence calibration:** overconfident in the 0.4–0.8 range, which holds under 3% of edges.
- **Machine speed:** the GPU looked thermally or power-limited during recent runs (timings roughly doubled).

## Next concrete step
Milestone 4 (v3 §6):
1. `docs/GRAPHML_SCHEMA.md` and a versioned schema check. XSD validation and round trip already exist.
2. DEXPI: fetch the current DEXPI specification and Proteus schema, and record the version.
3. `export/dexpi_mapping.yaml` with an explicit unmapped list, an `lxml` writer, then XSD or required-element validation.
4. Mapping coverage table.
