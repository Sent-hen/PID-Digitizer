# Error analysis (Milestone 5, final test predictions)

## Method

- **Inputs:** run `final_test` (detector `detector_m2c.pt`, sha1 `531ab26ce150`, the final pipeline).
- **Counts:** from `scripts/error_analysis.py`, stored in `error_analysis_counts.json`.
- **Matching:** the primary metric's (greedy IoU ≥ 0.5). Symbol categories match symbols class-agnostically.
- **Examples:** sampled at random (seed 0) in `docs/error_examples/`. Ground truth is drawn thick red and predictions thin blue; overlapping lines appear dark red. The yellow circle marks the error.
- **Tables:** all numbers are also in `docs/FINAL_RESULTS.md`.

## Counts

| category | Dataset-PID (75) | Synthetic (38) | OPEN100 (8) |
|---|---|---|---|
| GT symbols / GT edges | 8 947 / 38 009 | 869 / 5 052 | 975 / 3 111 |
| missed symbol | 9 (0.1%) | 0 | 46 (4.7%) |
| class confusion | 0 | 12 (1.4%) | 49 (5.0%) |
| false symbol | 39 | 73 | 209 |
| broken pipe (GT edge missed) | 3 220 (8.5%) | 405 (8.0%) | 1 853 (59.6%) |
| … of which node offset (pipe found, node 2–4 px off) | **2 574 (80% of misses)** | 247 (61%) | 332 (18%) |
| spurious link (predicted edge not in GT) | 3 220 | 460 | 805 |
| dashed-line error (non-solid missed or wrong type) | 881 (of 1 357 non-solid) | 166 (of 801) | 1 (none in GT) |

Dataset-PID's predicted edge total equals its GT edge total (38 009) by coincidence. The per-sheet counts differ, but false positives and false negatives sum to the same 3 220.

## Categories

### Node offset: the pipe is found, but a node misses by a few pixels (dominant on the generated subsets)
Examples: `broken_pipe_node_offset_1–4.jpg`, `broken_pipe_1–2.jpg`, `spurious_link_1–4.jpg`.

- **What happens:** the predicted pipe lies on the GT pipe, but a crossing or ankle is 2–4 px off. The 8×8 node then fails IoU 0.5, which tolerates about 2.7 px per axis. The edge is counted both missed and spurious, which is why the broken and spurious counts are close on Dataset-PID.
- **Causes:**
  1. **Annotation convention:** Dataset-PID GT marks crossings at the top-left edge of the stroke, while the line stage marks the stroke centre (PROGRESS, open issues). That's up to 2.5 px on 6 px strokes.
  2. **Ankle placement:** ankles sit on the *detected* box border, so box localization error moves them (mAP@0.5:0.95 is 0.95, not 1.0).
- **Fixes tried:** removing rot90/flip augmentation restored box precision (R-007).
- **Not applied, by design:** a per-subset convention shift would need the subset identity at inference, which would be leakage.
- **Effect:** these are scoring-tolerance errors, not topology errors. With a looser node match the generated-subset connectivity would be much higher, but the official IoU-0.5 protocol is kept.

### Broken pipe: the line is really not recovered (OPEN100)
Examples: `broken_pipe_3–4.jpg`, `missed_symbol_2.jpg`.

- **Share:** only 18% of OPEN100 misses are node offsets; the rest are real.
- **Causes:**
  1. **Crossing convention mismatch:** OPEN100 GT keeps most line crossings *unconnected* (only 2 four-way crossings in the whole subset), but the pipeline connects every H/V meeting, the Dataset-PID/Synthetic convention (D-007). An X-crossing therefore produces wrong junctions (`broken_pipe_4.jpg`).
  2. **Short nozzle stubs** below the 30 px minimum line length.
  3. **Symbols nested** inside a large symbol's box (e.g. internals of a vessel, the fans of an air cooler), blanked with the outer box.
  4. **Diagonal lines** (strict xfail, D-021).
  5. **Missed or misplaced detections:** no symbol means no ports.
- **Fixes tried:**
  - lighter ink threshold (+0.03 F1 on OPEN100 train, D-021);
  - cropping, which is worth +0.044 edge F1 on OPEN100 test (ablation).
- **Not fixed:** a per-sheet crossing convention would need a junction-dot / line-hop detector, which the GT can't train (no junction labels).

### Spurious link
Examples: `spurious_link_1–4.jpg`.

- **What happens:** mostly the mirror image of node offsets. The rest are dashed lines bridged across a junction gap (the 20 px dashed tolerance) and ports created near short text such as `4"`.
- **Ablation:** the ankle-port rule removes many text/outline ports on Dataset-PID (−0.014 F1 without it) but *costs* OPEN100 0.017 F1. OPEN100 GT contains ankle-ankle pairs on the same symbol that the rule deletes. This was found on test, so it's reported, not retuned.

### Dashed-line errors (Dataset-PID)
Examples: `dashed_error_1–4.jpg`.

- **Size:** 881 of 1 357 non-solid GT edges are missed or mistyped.
- **Causes:**
  1. Many sit next to junctions, where a dashed run meets a solid run and the merged segment takes the majority style.
  2. Dataset-PID's generator switches style along one polyline (the paper notes this), so GT non-solid edges can look solid.
  3. Node offsets, as above.
- **Fixes tried:** gap-sized join tolerance for dashed segments. Non-solid recall went 0.78 → 0.90 on oracle val, and the ablation shows −0.041 edge mAP on Dataset-PID test without it.

### False symbols
Examples: `false_symbol_1–4.jpg`.

- **OPEN100 (209):**
  - sheet-border zone markers (the triangles next to zone letters) detected as `arrow`;
  - duplicate or offset boxes on off-page connector labels;
  - small text boxes.
- **Synthetic (73):** offset second detections on large symbols.
- **Causes:** only 4 OPEN100 sheets in training (D-014), and border furniture that the generated subsets don't have. The crop removes the frame but not the zone-marker strip outside it.

### Missed symbols
Examples: `missed_symbol_1–4.jpg`.

- **Dataset-PID:** almost none (9), all tiny valves.
- **OPEN100 (46):**
  - small symbols in cluttered areas (instrument bubbles crowded together);
  - symbols on tank walls;
  - note boxes the annotators labelled as `general`.
- **Pump:** still has 0 AP on OPEN100 (only 5 training instances). See the per-class table in `FINAL_RESULTS.md`.

### Class confusion (OPEN100, Synthetic)
Examples: `class_confusion_1–4.jpg`.

- **OPEN100:** pump and heater outlines predicted as `general`/`tank`, instrument bubbles with unusual inner labels, flow elements.
- **Synthetic:** flag-shaped symbols confused between `general` and `arrow`.
- **Cause:** the coarse 7-class taxonomy merges very different drawings into `general`, which the paper also reports (the class with the highest confusion).

## What this means

| subset | what limits connectivity |
|---|---|
| Dataset-PID | Topology is essentially solved (oracle-symbol edge F1 0.954 test). The remaining loss is node placement within the IoU tolerance, plus dashed-line styling. |
| Synthetic | The same, with node-offset at 61% of misses and the detector costing 0.065 F1 vs oracle symbols. |
| OPEN100 (real) | The line stage, at oracle-symbol F1 0.594, and detection (−0.108 F1 vs oracle) are both far from solved. The crossing convention, nested and short connections, and only 4 real training sheets are the main reasons. Results on 8 test sheets carry high variance. |
