# P&ID Digitizer

Converts P&ID drawings into a graph, with equipment, valves and instruments as nodes and pipes and signal lines as edges. Outputs JSON and DEXPI-structured XML. Evaluated on [PID2Graph](https://zenodo.org/records/14803338) with graph-level metrics.

> **Status: Milestone 0 of 6.** The dataset audit, evaluation harness and scaffolding are done; there is no model yet. See [PROGRESS.md](PROGRESS.md) and the plan in [docs/PROPOSAL.md](docs/PROPOSAL.md).

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
```

On Linux or macOS, `make check`, `make inspect`, `make splits` and `make eval-oracle` wrap these commands.

## License and attribution

- **Code:** to be decided (no AGPL dependencies are used).
- **Dataset and derived data:** the PID2Graph dataset, and derived files in `splits/` and `docs/dataset_samples/`, are CC BY-SA 4.0. They are © the PID2Graph authors and upstream sources (Dataset-P&ID by Paliwal et al.; OPEN100 by the Energy Impact Center; symbols by S. B. Sandoval). See DATASET_NOTES.md.
