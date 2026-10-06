# P&ID Digitizer

Converts raster P&ID drawings into a graph, with equipment, valves and instruments as nodes and pipes and signal lines as edges. Outputs GraphML (primary, [schema](docs/GRAPHML_SCHEMA.md)) and DEXPI-structured XML (DEXPI 2.0.1, [coverage](docs/DEXPI_COVERAGE.md)). Evaluated on [PID2Graph](https://zenodo.org/records/14803338) with graph-level metrics.

> **Status: Milestones 0–4 of 6 complete.** Final test-set evaluation (Milestone 5) is pending; the table below is the Milestone 1 baseline. See [PROGRESS.md](PROGRESS.md) and the plan in [docs/PROPOSAL.md](docs/PROPOSAL.md).

## Baseline results (Milestone 1, test splits)

Our own splits are used, because PID2Graph releases no training data (see [DATASET_NOTES.md](DATASET_NOTES.md)). Predictions are GraphML files. All numbers are from [RESULTS.md](RESULTS.md) (R-003).

| test set | Symbols mAP@0.5 | Nodes AP@0.5 | Edge F1 | Edge mAP (PID2Graph protocol) |
|---|---|---|---|---|
| Dataset-PID (75 sheets) | 0.997 | 0.857 | 0.893 | 0.673 |
| PID2Graph Synthetic (38) | 0.991 | 0.940 | 0.925 | 0.907 |
| OPEN100, real (8 test sheets; 4 others used in training) | 0.704 | 0.501 | 0.492 | 0.375 |

## Dataset

PID2Graph (Stürmer, Graumann & Koch, IEEE DSAA 2025), CC BY-SA 4.0. Download it from Zenodo and unzip it so that `PID2Graph/Complete/...` sits at the repo root.

The format, quirks, official protocol and published baselines are in [DATASET_NOTES.md](DATASET_NOTES.md), and statistics are in [docs/dataset_stats.md](docs/dataset_stats.md).

## Developer setup

```bash
uv sync
uv run pytest                      # CPU unit tests (CI)
uv run pytest -m dataset           # needs PID2Graph on disk
uv run python scripts/inspect_dataset.py
uv run python scripts/make_splits.py
uv run python scripts/evaluate.py --oracle --split dataset_pid_test
uv run python scripts/train_detector.py   # ~0.9 GPU-h on an RTX 4070 Laptop
uv run python scripts/run_baseline.py --run m1b_baseline --split dataset_pid_test --split synthetic_test --split open100_test
```

`make check`, `make train`, `make baseline`, `make baseline-val` and `make baseline-oracle` wrap these commands.

## License and attribution

- **Code:** to be decided (no AGPL dependencies are used).
- **Dataset and derived data:** the PID2Graph dataset, and derived files in `splits/` and `docs/dataset_samples/`, are CC BY-SA 4.0. They are © the PID2Graph authors and upstream sources (Dataset-P&ID by Paliwal et al.; OPEN100 by the Energy Impact Center; symbols by S. B. Sandoval). See DATASET_NOTES.md.
