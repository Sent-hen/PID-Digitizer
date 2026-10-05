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

## R-002 Milestone 1 baseline (walking skeleton): the numbers to beat

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
