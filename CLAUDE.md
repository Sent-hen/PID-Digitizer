# CLAUDE.md

Project spec: [docs/PROPOSAL.md](docs/PROPOSAL.md). Work milestone by milestone, and don't start a milestone until the previous one's acceptance criteria pass.

**Read first, every session:** `PROGRESS.md` (where we are, next step), then `DECISIONS.md` and `DATASET_NOTES.md`.

## Rules
- **Update `PROGRESS.md`** at the end of every session. Commit after each completed task, and never leave the repo broken.
- **Logs:**
  - Non-obvious choices go in `DECISIONS.md` (with the rejected alternative).
  - Every reported number goes in `RESULTS.md` (config, commit, hardware). Never quote a number that isn't there.
- **Config:** all paths and hyperparameters live in `configs/*.yaml`. Seeds are fixed. Split ID lists are committed; data is not.
- **Never commit** `PID2Graph/`, weights over 50 MB, caches (`.cache/`) or generated outputs (`outputs/`).
- **Tests:** CI is CPU-only and uses fixtures in `tests/data/` only. Mark tests that need the dataset with `@pytest.mark.dataset` and slow ones with `@pytest.mark.slow`.
- **Licenses:** no AGPL dependencies (no Ultralytics). Record every model library's license in DECISIONS.md.
- **Ask first** before:
  - more than ~3 GPU-hours on one experiment;
  - adding a heavy dependency or service;
  - changing the output schema after Milestone 4;
  - any change to scope.

## Commands (`make check|baseline|baseline-val|baseline-oracle|train` wrap these; make is installed via Chocolatey)
```
.venv/Scripts/uv sync                                  # install (UV_PROJECT_ENVIRONMENT=.venv)
.venv/Scripts/python -m pytest                         # unit tests (CI set)
.venv/Scripts/python -m pytest -m dataset              # tests needing PID2Graph on disk
.venv/Scripts/ruff check src tests scripts
.venv/Scripts/mypy
.venv/Scripts/python scripts/inspect_dataset.py        # stats + overlays -> docs/
.venv/Scripts/python scripts/make_splits.py            # splits/ (cached fingerprints in .cache/)
.venv/Scripts/python scripts/evaluate.py --oracle --split dataset_pid_test
```

## Layout notes
- Canonical model: `src/pid/io/drawing.py`. Everything converts to and from `Drawing`.
- Class names are canonical (`ankle` = raw `connector`, `inlet_outlet` = raw `inlet/outlet`). The mapping is in `configs/data.yaml`.
- Drawing uid = `<subset_key>/<id>`, e.g. `dataset_pid/17`. Predictions go in `<pred_dir>/<subset_key>/<id>.json`.
