# Results

Every number reported anywhere in this repo must appear here, with the config, commit, hardware and command that produced it. Raw records are appended to `outputs/eval_log.jsonl` by `scripts/evaluate.py`.

Hardware: see DECISIONS.md D-001.

## R-000 Harness oracle (ground truth vs itself)

Sanity check, not a result. Command: `python scripts/evaluate.py --oracle --split dataset_pid_test --split synthetic_test --split open100_test`. Config `configs/data.yaml`, commit `70a6dc5` + Milestone 0 working tree.

| split | drawings | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP@0.5 | Edge P / R / F1 | Edge mAP (PID2Graph) |
|---|---|---|---|---|---|---|
| dataset_pid_test | 75 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 |
| synthetic_test | 38 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 |
| open100_test | 12 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1.0 |

Perturbation tests (`tests/test_oracle.py`) confirm that deleting k edges, adding spurious edges, dropping nodes, shifting boxes below IoU 0.5, flipping line types and relabelling classes change each metric by exactly the expected amount.

## R-001 Oracle-symbol diagnostic (line + graph stages only)

The Milestone 1 pipeline with **ground-truth symbol boxes** replacing the detector, so only line extraction and graph assembly are measured. This is the "oracle detection" half of the error-propagation analysis, not a baseline result.
- **Command:** `python scripts/run_baseline.py --run m1_oracle_symbols --oracle-symbols --split ...`
- **Config:** `configs/pipeline.yaml`, `configs/data.yaml`.
- **Commit:** `df6fec2`. CPU only.

| split | n | Nodes AP@0.5 | Edge P / R / F1 | per-drawing F1 median / p10 | line-type acc | Edge mAP (PID2Graph) |
|---|---|---|---|---|---|---|
| dataset_pid_val | 75 | 0.910 | 0.918 / 0.954 / 0.935 | 0.934 / 0.890 | 0.976 | 0.685 |
| synthetic_val | 38 | 0.984 | 0.969 / 0.982 / 0.975 | 0.978 / 0.944 | 0.974 | 0.917 |
| dataset_pid_test | 75 | 0.917 | 0.915 / 0.952 / 0.933 | 0.939 / 0.852 | 0.980 | 0.684 |
| synthetic_test | 38 | 0.984 | 0.975 / 0.987 / 0.981 | 0.979 / 0.953 | 0.975 | 0.921 |
| open100_test | 12 | 0.660 | 0.816 / 0.544 / 0.653 | 0.649 / 0.558 | 0.998 | 0.509 |

Symbol mAP is 1.0 by construction (GT boxes). Zero failures. Runtime is 0.3–0.5 s per drawing, excluding detection.

What the numbers show:
- **Dataset-PID edge mAP** is pulled down by dashed lines: non-solid AP is 0.44 against 0.93 for solid (val).
- **OPEN100's low recall** comes from diagonal lines, direct symbol–symbol edges and faint signal lines (DECISIONS D-010).

## R-002 Milestone 1 baseline, original split (superseded by R-003)

This was the first run, with OPEN100 fully held out (12 sheets). After the proposal v3 change to option B, R-003 is the baseline to beat; R-002 is kept for the record.

Pipeline:
1. rescale the sheet to a 7000 px long side;
2. Faster R-CNN R50-FPN v2 on 1024² tiles with 256 px overlap;
3. complete-first IoS NMS merge;
4. orthogonal-morphology lines;
5. graph assembly.

No OCR and no confidence model: structural nodes and edges get score 1.0, symbols keep their detector score.

**Detector training:**
- **Config:** `configs/detect.yaml`.
- **Data:** 19 398 tiles from `dataset_pid_train` + `synthetic_train` (524 sheets). No overlap with val/test was verified against the tile index.
- **Schedule:** 6000 iterations, batch 4, SGD lr 0.01 with cosine schedule, AMP.
- **Run:** run id `20261004-213543`, commit `df6fec2`. Final loss 0.030.
- **Time:** 3138 s, 0.87 GPU-hours.
- **Hardware:** RTX 4070 Laptop 8 GB, torch 2.14.1+cu132 (D-001, D-012).

**Evaluation:**
- **Command:** `make baseline RUN=m1_baseline` (test) and `make baseline-val RUN=m1_baseline` (val).
- **Config:** `configs/pipeline.yaml`. Score threshold 0.3.
- **Scope:** run once on test, with no tuning afterwards. Zero failures on every split.
- **Speed:** about 3.1–3.5 s per drawing end to end (GPU detection plus CPU lines/graph).

| split | n | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP@0.5 | Edge P / R / F1 | per-drawing F1 median / p10 | line-type acc | Edge mAP (PID2Graph) |
|---|---|---|---|---|---|---|---|---|
| dataset_pid_val | 75 | 1.000 | 0.953 | 0.878 | 0.888 / 0.924 / 0.906 | 0.898 / 0.837 | 0.976 | 0.685 |
| synthetic_val | 38 | 0.998 | 0.904 | 0.939 | 0.916 / 0.930 / 0.923 | 0.922 / 0.866 | 0.974 | 0.903 |
| **dataset_pid_test** | 75 | 0.997 | 0.953 | 0.868 | 0.885 / 0.922 / 0.903 | 0.907 / 0.798 | 0.980 | 0.684 |
| **synthetic_test** | 38 | 0.993 | 0.913 | 0.942 | 0.927 / 0.936 / 0.932 | 0.927 / 0.900 | 0.974 | 0.908 |
| **open100_test** (real, held out) | 12 | 0.416 | 0.262 | 0.374 | 0.424 / 0.394 / 0.409 | 0.421 / 0.260 | 0.996 | 0.369 |

**Per-class AP@0.5 (test):**

| class | dataset_pid | synthetic | open100 |
|---|---|---|---|
| general | 0.988 | 0.999 | 0.506 |
| tank | – | 0.966 | 0.015 |
| valve | 1.000 | 1.000 | 0.777 |
| instrumentation | 1.000 | – | 0.980 |
| pump | – | 1.000 | 0.000 |
| arrow | 1.000 | 1.000 | 0.632 |
| inlet_outlet | – | – | 0.000 (no training data, D-007) |

"–" means the class is absent from that subset.

**Edge AP by line type (test):**

| subset | solid | non-solid |
|---|---|---|
| dataset_pid | 0.934 | 0.434 |
| synthetic | 0.928 | 0.888 |
| open100 | 0.369 | n/a (OPEN100 has no non-solid lines) |

**Error propagation (R-001 vs R-002, edge F1 on test):**

| subset | oracle symbols | predicted symbols | gap |
|---|---|---|---|
| dataset_pid | 0.933 | 0.903 | −0.030 |
| synthetic | 0.981 | 0.932 | −0.049 |
| open100 | 0.653 | 0.409 | −0.244 |

On the generated subsets, most remaining error is in lines and graph assembly. On real drawings, detection is the dominant loss, followed by line extraction (oracle 0.653).

**Context: paper Table III (stitched).** This is *not comparable*: their training data is different and unreleased, and their test sets are the full subsets (DATASET_NOTES).

| | Symbols mAP | Nodes AP | Edge mAP |
|---|---|---|---|
| Their modular baseline, OPEN100 | 76.99 | 52.14 | 45.89 |
| Ours, OPEN100 | 41.6 | 37.4 | 36.9 |
| Their modular baseline, Dataset-P&ID | 83.95 | 89.32 | 85.46 |
| Ours, Dataset-P&ID | 99.7 | 86.8 | 68.4 |

Caveat for Dataset-P&ID: the paper merged all edges into a single class, while we score solid and non-solid separately, and our non-solid AP is 0.43.

Near-perfect symbol mAP on the generated subsets reflects that their symbols are rendered from a small fixed template set, so train and test symbols are visually near-identical. OPEN100 is the meaningful number.

## R-003 Milestone 1 baseline, proposal v3 (option B, GraphML output): **the numbers to beat**

Same pipeline as R-002, with three differences:
- 4 OPEN100 sheets (0, 6, 7, 9) were added to training, with their tiles repeated 3× (D-014);
- the OPEN100 test now has 8 sheets;
- predictions are written as GraphML and each file is validated against the official XSD and reloaded (D-015).

**Detector training:**
- **Config:** `configs/detect.yaml`.
- **Data:** 19 857 tile records (13 973 Dataset-PID, 5 425 Synthetic, 459 OPEN100 = 153 unique × 3).
- **Schedule:** 6000 iterations, batch 4, AMP. Final loss 0.067.
- **Run:** run id `20261004-230530`.
- **Commit:** `54146c5`, plus the option B config, committed in `775d43a`.
- **Time:** 3039 s, 0.84 GPU-hours.
- **Hardware:** RTX 4070 Laptop, torch 2.14.1+cu132.
- **Weights:** `weights/detector_m1b.pt`.

**Evaluation:**
- **Command:** `make baseline RUN=m1b_baseline` (test) and `make baseline-val RUN=m1b_baseline` (val).
- **Config:** `configs/pipeline.yaml`. Score threshold 0.3.
- **Scope:** test run once, with no tuning afterwards.
- **Outcome:** zero failures and **zero invalid GraphML files** on every split.

| split | n | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP@0.5 | Edge P / R / F1 | per-drawing F1 median / p10 | line-type acc | Edge mAP (PID2Graph) |
|---|---|---|---|---|---|---|---|---|
| dataset_pid_val | 75 | 1.000 | 0.950 | 0.847 | 0.854 / 0.919 / 0.886 | 0.888 / 0.810 | 0.976 | 0.647 |
| synthetic_val | 38 | 0.996 | 0.906 | 0.927 | 0.910 / 0.921 / 0.915 | 0.916 / 0.869 | 0.973 | 0.904 |
| **dataset_pid_test** | 75 | 0.997 | 0.946 | 0.857 | 0.871 / 0.917 / 0.893 | 0.904 / 0.799 | 0.980 | 0.673 |
| **synthetic_test** | 38 | 0.991 | 0.919 | 0.940 | 0.919 / 0.930 / 0.925 | 0.926 / 0.879 | 0.974 | 0.907 |
| **open100_test** (8 real sheets) | 8 | 0.704 | 0.500 | 0.501 | 0.571 / 0.433 / 0.492 | 0.515 / 0.401 | 0.996 | 0.375 |

**Per-class AP@0.5 (test):**

| class | dataset_pid | synthetic | open100 |
|---|---|---|---|
| general | 0.989 | 0.990 | 0.807 |
| tank | – | 0.967 | 0.356 |
| valve | 1.000 | 1.000 | 0.857 |
| instrumentation | 1.000 | – | 0.989 |
| pump | – | 1.000 | 0.000 |
| arrow | 1.000 | 1.000 | 0.926 |
| inlet_outlet | – | – | 0.995 |

"–" means the class is absent from that subset.

**Edge AP by line type (test):**

| subset | solid | non-solid |
|---|---|---|
| dataset_pid | 0.923 | 0.422 |
| synthetic | 0.922 | 0.892 |
| open100 | 0.375 | n/a (OPEN100 has no non-solid lines) |

**Effect of option B on the same 8 OPEN100 test sheets.** Both rows use identical sheets: R-002's saved predictions were re-scored on the 8-sheet list.

| detector | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP | Edge P / R / F1 | Edge mAP | tank / pump / inlet_outlet AP |
|---|---|---|---|---|---|---|
| R-002, no OPEN100 in training | 0.397 | 0.249 | 0.366 | 0.425 / 0.370 / 0.395 | 0.336 | 0.020 / 0.000 / 0.000 |
| R-003, option B | **0.704** | **0.500** | **0.501** | **0.571 / 0.433 / 0.492** | **0.375** | 0.356 / 0.000 / 0.995 |

- **Cost on the generated subsets:** small. Dataset-PID test edge F1 went from 0.903 to 0.893, and Synthetic from 0.932 to 0.925.
- **Pump** stays at 0 AP: there are only 5 training instances, all from sheet 7.

**Oracle symbols on the 8-sheet OPEN100 test:**
- **Run:** `m1b_oracle_symbols`, same line/graph code as R-001.
- **Result:** node AP 0.652, edge P/R/F1 0.799 / 0.505 / 0.619, edge mAP 0.464.
- **Error propagation on OPEN100:** edge F1 drops from 0.619 with GT symbols to 0.492 with predicted symbols, a gap of 0.127.
- **Other subsets:** Dataset-PID and Synthetic oracle numbers are unchanged from R-001, since their splits are identical.

## R-004 Milestone 2: border cropping (val)

**Crop detector** (`preprocess/crop.py`, `configs/pipeline.yaml: crop`):
- **Scope:** content box vs. the GT box derived from ignore regions, on train and val sheets.
- **Result:**
  - Dataset-PID train/val: mean IoU 0.950, min 0.949. The ceiling is about 0.95, because the GT box is a fixed inset rectangle (D-017).
  - OPEN100 train: mean IoU 0.959, min 0.937.
  - Synthetic train/val: 0 of 212 sheets cropped (no frame).
  - Zero GT nodes fall outside a predicted crop on any train/val sheet.
- **OPEN100 test** sheets weren't inspected.

**Effect on the full pipeline (val):**
- **Detector:** `weights/detector_m1b.pt` (sha1 `b566b21f84ad`).
- **Runs:** `m1b_baseline` (crop off, R-003) vs `m1b_crop_clean` (crop on).
- **Commit:** `b64644e`.

| split | crop | Nodes AP | Edge P / R / F1 | median F1 | Edge mAP |
|---|---|---|---|---|---|
| dataset_pid_val | off | 0.847 | 0.854 / 0.919 / 0.886 | 0.888 | 0.647 |
| dataset_pid_val | **on** | **0.869** | **0.889 / 0.919 / 0.904** | **0.902** | **0.686** |
| synthetic_val | off | 0.927 | 0.910 / 0.921 / 0.915 | 0.916 | 0.904 |
| synthetic_val | on | 0.927 | 0.910 / 0.921 / 0.915 | 0.916 | 0.904 |

Cropping is now on by default. Recall is unchanged and precision rises, because title-block and frame false positives are removed. Synthetic is identical because no Synthetic sheet is cropped.

## R-005 Milestone 2: detector with augmentation + repeat-factor sampling (val, clean)

**Training:**
- **Config:** `configs/detect.yaml`, with rot90 + flip, `degrade_prob` 0.5 and RFS threshold 0.04 (D-018).
- **Run:** run id `20261005-130438`, commit `00bb6eb` + the augmentation code (committed `b64644e`).
- **Schedule:** 6000 iterations. Final loss 0.064.
- **Time:** 3167 s, 0.88 GPU-hours.
- **Weights:** `weights/detector_m2.pt` (sha1 `77e0c127b746`).

**Evaluation:** crop on; same pipeline as R-004.

| split | detector | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP | Edge P / R / F1 | median F1 | Edge mAP |
|---|---|---|---|---|---|---|---|
| dataset_pid_val | m1b | 1.000 | 0.950 | 0.869 | 0.889 / 0.919 / 0.904 | 0.902 | 0.686 |
| dataset_pid_val | m2 | 0.998 | 0.867 | 0.747 | 0.772 / 0.798 / 0.785 | 0.762 | 0.685 |
| synthetic_val | m1b | 0.996 | 0.906 | 0.927 | 0.910 / 0.921 / 0.915 | 0.916 | 0.904 |
| synthetic_val | m2 | 0.993 | 0.877 | 0.910 | 0.891 / 0.899 / 0.895 | 0.901 | 0.901 |

**On clean val, m2 is worse.** Detection at IoU 0.5 is unchanged, but localization (mAP@0.5:0.95) drops by 0.03–0.08, and that propagates strongly to connectivity.

The reason is the line stage: it places ankles exactly on symbol bbox borders, and an 8 px ankle tolerates only about 2.7 px of error at IoU 0.5. So box precision, not detection, is what limits edge F1. Robustness on degraded val is evaluated next (R-006).

## R-006 Milestone 2: robustness at degradation level 2, deskew, augmentation variants (val)

**Setup:**
- **Degradation:** `configs/degrade.yaml` level 2, applied to the original sheet; GT mapped through the skew (D-016).
- **Pipeline:** crop on. Each row records its detector weights hash in its metrics file.
- **Commit:** `e749f73`.

**Detector weights:**

| model | weights sha1 | augmentation |
|---|---|---|
| m1b | `b566b21f84ad` | none (R-003) |
| m2 | `77e0c127b746` | rot90 + flip, degrade p = 0.5 with skew, RFS (R-005) |
| m2b | `b844ff9084ce` | rot90 + flip, degrade p = 0.3 without skew, RFS. Run `20261005-143151`, 3055 s. |

| split | detector | deskew | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP | Edge P / R / F1 | Edge mAP |
|---|---|---|---|---|---|---|---|
| dataset_pid_val | m1b | off | 1.000 | 0.908 | 0.488 | 0.524 / 0.470 / 0.496 | 0.507 |
| dataset_pid_val | m2 | off | 1.000 | 0.868 | 0.397 | 0.418 / 0.372 / 0.394 | 0.517 |
| dataset_pid_val | m1b | **on** | 1.000 | 0.920 | **0.866** | **0.881 / 0.909 / 0.895** | **0.680** |
| dataset_pid_val | m2b | on | 1.000 | 0.874 | 0.762 | 0.784 / 0.803 / 0.793 | 0.673 |
| synthetic_val | m1b | off | 0.858 | 0.733 | 0.391 | 0.487 / 0.320 / 0.386 | 0.535 |
| synthetic_val | m2 | off | 0.993 | 0.876 | 0.447 | 0.562 / 0.379 / 0.453 | 0.574 |
| synthetic_val | m1b | **on** | 0.859 | 0.712 | 0.770 | 0.749 / 0.745 / 0.747 | 0.833 |
| synthetic_val | m2b | on | **0.992** | **0.875** | **0.906** | **0.882 / 0.889 / 0.886** | **0.892** |

**m2b on clean val:**

| split | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP | Edge P / R / F1 | Edge mAP |
|---|---|---|---|---|---|
| dataset_pid_val | 1.000 | 0.871 | 0.748 | 0.774 / 0.799 / 0.786 | 0.683 |
| synthetic_val | 0.993 | 0.889 | 0.923 | 0.896 / 0.906 / 0.901 | 0.902 |

Findings:
1. **Deskew** restores the line stage under skew. At level 2, Dataset-PID edge F1 goes from 0.496 to 0.895 (clean: 0.904), and Synthetic from 0.386 to 0.747. On clean sheets the estimated angle is at most 0.01°, below the 0.05° threshold, so they're never resampled. Deskew is now enabled by default.
2. **Degradation augmentation** makes detection robust: Synthetic symbol mAP at level 2 is 0.99 against 0.86 without it.
3. **Clean Dataset-PID localization** (mAP@0.5:0.95 0.95 → 0.87, edge F1 0.904 → 0.786) is *not* caused by skewed training targets, since m2b without skew shows the same loss. The remaining suspect is rot90/flip versus the asymmetric Dataset-PID box convention: border-to-ink gap p90 is 5 px on the left/top and 3 px on the right/bottom. The m2c variant, without rot/flip, tests this.

## R-007 Milestone 2: adopted detector m2c (no rot/flip) + overlap sweep (val)

**m2c training:**
- **Augmentation:** degrade p = 0.3 without skew, RFS 0.04, **no rot90/flip**.
- **Run:** run id `20261005-155321`, commit `e749f73` + config.
- **Schedule:** 6000 iterations. Final loss 0.025.
- **Time:** 6169 s, 1.7 GPU-hours. The machine ran at about half speed, the GPU apparently thermally or power-limited at 79 °C and 18 W, so wall-clock timings from this run onward aren't comparable with earlier rows. Accuracy is unaffected.
- **Weights:** `weights/detector_m2c.pt` (sha1 `531ab26ce150`).

**Evaluation:**
- **Pipeline:** crop and deskew on (deskew doesn't resample clean sheets).
- **Commit:** `e749f73` + the `--set-detect` option.

| split | detector | input | Symbols mAP@0.5 | mAP@0.5:0.95 | Nodes AP | Edge P / R / F1 | median F1 | Edge mAP |
|---|---|---|---|---|---|---|---|---|
| dataset_pid_val | m1b (R-004) | clean | 1.000 | 0.950 | 0.869 | 0.889 / 0.919 / 0.904 | 0.902 | 0.686 |
| dataset_pid_val | **m2c** | clean | 1.000 | 0.950 | 0.869 | 0.887 / 0.916 / 0.901 | 0.896 | 0.687 |
| synthetic_val | m1b (R-004) | clean | 0.996 | 0.906 | 0.927 | 0.910 / 0.921 / 0.915 | 0.916 | 0.904 |
| synthetic_val | **m2c** | clean | 0.996 | 0.878 | 0.925 | 0.900 / 0.911 / 0.905 | 0.895 | 0.896 |
| dataset_pid_val | m1b (R-006) | level 2 | 1.000 | 0.920 | 0.866 | 0.881 / 0.909 / 0.895 | 0.883 | 0.680 |
| dataset_pid_val | **m2c** | level 2 | 1.000 | 0.946 | 0.881 | 0.893 / 0.922 / 0.907 | 0.900 | 0.681 |
| synthetic_val | m1b (R-006) | level 2 | 0.859 | 0.712 | 0.770 | 0.749 / 0.745 / 0.747 | 0.751 | 0.833 |
| synthetic_val | **m2c** | level 2 | 0.996 | 0.871 | 0.904 | 0.873 / 0.881 / 0.877 | 0.887 | 0.893 |

**Per-class AP@0.5 (val, m2c):**

| class | Dataset-PID clean | Dataset-PID L2 | Synthetic clean | Synthetic L2 | Synthetic L2, m1b for contrast |
|---|---|---|---|---|---|
| general | 1.000 | 1.000 | 0.990 | 0.989 | 0.947 |
| tank | – | – | 0.992 | 0.992 | 0.982 |
| valve | 1.000 | 1.000 | 1.000 | 1.000 | 0.595 |
| instrumentation | 1.000 | 1.000 | – | – | – |
| pump | – | – | 1.000 | 1.000 | 0.772 |
| arrow | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

inlet_outlet exists only in OPEN100, which has no val split (D-014).

**Removing rot/flip** restores clean Dataset-PID localization (mAP@0.5:0.95 0.871 → 0.950, edge F1 0.786 → 0.901). This confirms that rot/flip conflicted with the asymmetric Dataset-PID box convention.

**Inference tile overlap 512 vs 256** (m2c, clean val, `--set-detect tiles.overlap=512`):

| split | overlap | Edge F1 | Nodes AP |
|---|---|---|---|
| dataset_pid_val | 256 | 0.901 | 0.869 |
| dataset_pid_val | 512 | 0.899 | 0.867 |
| synthetic_val | 256 | 0.905 | 0.925 |
| synthetic_val | 512 | 0.908 | 0.927 |

Overlap 512 changes metrics by ≤0.003 and doubles detection time, so 256 is kept. Tile *size* wasn't swept: it needs a retrain per value, and val symbol mAP@0.5 is already 0.996–1.000. The open question for size is large OPEN100 tanks, which val can't measure.

**Milestone 2 val summary** (vs the M1 baseline R-003, crop off, no deskew, m1b):

| split | input | Nodes AP | Edge F1 |
|---|---|---|---|
| Dataset-PID | clean | 0.847 → 0.869 | 0.886 → 0.901 |
| Synthetic | clean | 0.927 → 0.925 | 0.915 → 0.905 |
| Dataset-PID | level 2 | 0.488 → 0.881 | 0.496 → 0.907 |
| Synthetic | level 2 | 0.391 → 0.904 | 0.386 → 0.877 |

Clean symbol mAP@0.5 is saturated at 0.996–1.000 for both models.

## R-008 Milestone 3: lines, connectivity, confidence (val)

**Pipeline:**
- **Detector:** m2c (sha1 `531ab26ce150`). Deskew and crop on.
- **Changes in this milestone** (D-021):
  - ankle-port rule;
  - dashed join/snap tolerance 20 px;
  - ink threshold 180, capped at the paper's median gray level − 25;
  - calibrated confidence (`configs/confidence_model.json`).
- **Config hash:** `23bb6984b356`.
- **Commit:** `297c309` + the paper-relative threshold (committed with this entry).

**Confidence model fit** (`scripts/fit_confidence.py`, val, 43 660 predicted edges, 91.5% correct):
- 2-fold cross-validation by drawing, so both numbers are out-of-sample:
  - ECE = **0.015**;
  - edge mAP **0.787** with constant scores vs **0.905** with confidence (pooled over both val subsets).
- Reliability diagram: `docs/reliability_diagram.png`. Sample review overlay: `docs/confidence_overlay_dataset_pid_147.jpg`.
- The final model is fitted on all of val, so the val edge mAP rows below are in-sample for the confidence model. The CV numbers above and the test run are the out-of-sample checks.

**Full pipeline, val:**

| split | input | Symbols mAP@0.5 | Nodes AP | Edge P / R / F1 | median / p10 F1 | Edge mAP (AP solid / non-solid) |
|---|---|---|---|---|---|---|
| dataset_pid_val | clean | 1.000 | 0.935 | 0.916 / 0.917 / **0.917** | 0.920 / 0.867 | **0.909** (0.975 / 0.843) |
| synthetic_val | clean | 0.996 | 0.933 | 0.899 / 0.911 / 0.905 | 0.895 / 0.865 | **0.918** (0.925 / 0.910) |
| dataset_pid_val | level 2 | 1.000 | 0.936 | 0.907 / 0.902 / **0.904** | 0.902 / 0.857 | 0.894 (0.960 / 0.827) |
| synthetic_val | level 2 | 0.996 | 0.903 | 0.860 / 0.876 / 0.868 | 0.882 / 0.802 | 0.911 (0.903 / 0.920) |

Runs `m3_final_clean_v2`, `m3_final_L2_v2`. An earlier run with a fixed threshold of 180 (`m3_final_L2`) collapsed on degraded Dataset-PID, to edge F1 0.687 with p10 = 0, because gray paper at about 186 fell below the threshold. The paper-relative cap fixes this, and clean results are unchanged.

**Oracle symbols** (line + graph only, run `m3_final_oracle`, before the threshold cap; clean sheets are unaffected by the cap):

| split | Nodes AP | Edge P / R / F1 | Edge mAP |
|---|---|---|---|
| dataset_pid_val | 0.967 | 0.954 / 0.955 / 0.955 | 0.913 |
| synthetic_val | 0.985 | 0.967 / 0.982 / 0.974 | 0.935 |
| open100_train (4 sheets) | 0.729 | 0.910 / 0.595 / 0.719 | 0.590 |

**Milestone 3 vs the Milestone 1 baseline** (R-003 val, edge F1 / edge mAP):

| split | M1 (R-003) | M3 |
|---|---|---|
| Dataset-PID | 0.886 / 0.647 | **0.917 / 0.909** |
| Synthetic | 0.915 / 0.904 | 0.905 / **0.918** |

Synthetic F1 is −0.010 vs M1. That's attributable to the M2 detector choice (m2c trades 0.010 clean F1 for robustness, R-007). With oracle symbols, Synthetic edge F1 is unchanged at 0.974–0.975 through M3, so the line stage didn't regress (D-022).
