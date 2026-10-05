# P&ID Digitizer

Converts P&ID drawings into a graph, with equipment, valves and instruments as nodes and pipes and signal lines as edges. Outputs JSON and DEXPI-structured XML. Evaluated on [PID2Graph](https://zenodo.org/records/14803338) with graph-level metrics.

> **Status: Milestone 1 of 6 complete (walking-skeleton baseline).** See [PROGRESS.md](PROGRESS.md) and the plan in [docs/PROPOSAL.md](docs/PROPOSAL.md).

## Baseline results (Milestone 1, test splits)

Our own splits are used, because PID2Graph releases no training data (see [DATASET_NOTES.md](DATASET_NOTES.md)). All numbers are from [RESULTS.md](RESULTS.md) (R-002).

| test set | Symbols mAP@0.5 | Nodes AP@0.5 | Edge F1 | Edge mAP (PID2Graph protocol) |
|---|---|---|---|---|
| Dataset-PID (75 sheets) | 0.997 | 0.868 | 0.903 | 0.684 |
| PID2Graph Synthetic (38) | 0.993 | 0.942 | 0.932 | 0.908 |
| OPEN100, real, held out (12) | 0.416 | 0.374 | 0.409 | 0.369 |

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
uv run python scripts/run_baseline.py --run m1_baseline --split dataset_pid_test --split synthetic_test --split open100_test
```

`make check`, `make train`, `make baseline`, `make baseline-val` and `make baseline-oracle` wrap these commands.

## License and attribution

- **Code:** to be decided (no AGPL dependencies are used).
- **Dataset and derived data:** the PID2Graph dataset, and derived files in `splits/` and `docs/dataset_samples/`, are CC BY-SA 4.0. They are © the PID2Graph authors and upstream sources (Dataset-P&ID by Paliwal et al.; OPEN100 by the Energy Impact Center; symbols by S. B. Sandoval). See DATASET_NOTES.md.
