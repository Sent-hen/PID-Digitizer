# CLAUDE.md

Guide for continuing work on this project.
- **Spec:** [docs/PROPOSAL.md](docs/PROPOSAL.md) (v3), milestones 0–6 complete.
- **Read first, every session:** `PROGRESS.md` (state, open issues, next step), then `DECISIONS.md` (why things are the way they are, D-001 to D-025) and `DATASET_NOTES.md` (dataset quirks).

## Rules
- **Update `PROGRESS.md`** at the end of every session. Commit after each completed task, and never leave the repo broken. Work on a branch and open a PR to `main`.
- **Logs:**
  - Non-obvious choices go in `DECISIONS.md` (with the rejected alternative).
  - Every reported number goes in `RESULTS.md` (config, commit, weights sha1, hardware). Never quote a number that isn't there.
- **Test sets are run once per final configuration** (`scripts/final_eval.py` skips finished runs). Tune on `*_val` splits only. OPEN100 has no val split (D-014): judge real-drawing changes on `open100_train` with `--oracle-symbols`, and say so.
- **Config:** paths and hyperparameters go in `configs/*.yaml`. Seeds are fixed. Split ID lists are committed; data and weights are not.
- **Never commit** `PID2Graph/`, `weights/`, `.cache/` or `outputs/`. Weights are published as GitHub release assets (`scripts/fetch_weights.py`).
- **Tests:** CI is CPU-only and uses fixtures in `tests/data/`. Mark tests needing the dataset with `@pytest.mark.dataset` and slow ones with `@pytest.mark.slow`. API tests inject a fake pipeline.
- **Licenses:** no AGPL dependencies (no Ultralytics). Record each model library's license in DECISIONS.md. Dataset-derived files and weights are CC BY-SA 4.0 with attribution.
- **Ask the user first** before:
  - more than ~3 GPU-hours on one experiment;
  - adding a heavy dependency or service;
  - changing the GraphML schema (`docs/GRAPHML_SCHEMA.md`);
  - any change to scope;
  - publishing anything (releases, weights).

## Commands
`make` is installed (Chocolatey). The venv is `.venv` (Python 3.13, torch 2.14.1+cu132); set `UV_PROJECT_ENVIRONMENT=.venv` for uv.
```
make check            # ruff + mypy --strict + CPU tests
make test-dataset     # tests that need PID2Graph/ on disk
make weights          # fetch detector weights (sha1-checked) from the GitHub release
make serve            # API + review UI on http://localhost:8000 (uses the GPU locally)
make demo             # docker compose up --build + upload the bundled sample drawing
make train            # detector training (configs/detect.yaml; ~1-2 GPU-h)
make baseline-val RUN=<name>     # full pipeline on val splits -> outputs/preds/<name>/
make eval / make report          # final test matrix (once) / regenerate all tables
.venv/Scripts/python scripts/run_baseline.py --run X --split dataset_pid_val [--oracle-symbols] [--degrade 2] [--set graph.join_tol=10] [--dexpi]
.venv/Scripts/python scripts/analyze_errors.py --pred-dir outputs/preds/X --split dataset_pid_val
.venv/Scripts/python scripts/fit_confidence.py --split dataset_pid_val --split synthetic_val   # after changing graph features
```

## Code map (`src/pid/`)
- **`io/`:**
  - `drawing.py`: the canonical `Drawing`. Everything converts to and from it.
  - `pid2graph.py`: GT loader. `safe_id` sanitizes OPEN100 node ids.
  - `splits.py`, `overlay.py` (`color_by="confidence"`).
- **`preprocess/`:** `tiling.py` (rescale to a 7000 px long side), `deskew.py`, `crop.py` (frame and title block), `degrade.py` (robustness simulator).
- **`detect/`:** `model.py` (Faster R-CNN, tiled inference), `merge.py` (tile-seam NMS), `tiles.py` (training cache), `augment.py`.
- **`lines/extract.py`:** orthogonal line segments; paper-relative ink threshold.
- **`graph/`:**
  - `assemble.py`: crossings, ankles as ports, the dashed-join tolerance, cleanup rules, and per-edge features;
  - `confidence.py`: logistic model in `configs/confidence_model.json`.
- **`pipeline.py`:** `digitize()` wires all of the above.
- **`export/`:** `graphml.py` (primary output, XSD-validated), `dexpi.py` + `dexpi_mapping.yaml` + `dexpi_xsd/` (DEXPI 2.0.1).
- **`eval/`:** matching, pycocotools mAP, edge P/R/F1, the official PID2Graph edge mAP.
- **`api/`:** `app.py` (FastAPI), `store.py` (SQLite jobs), `corrections.py` (typed reviewer edits).
- **`ui/`:** static review UI (vanilla JS canvas).

## Gotchas
- **Structural nodes:** 8×8 boxes at IoU 0.5, so a 3 px offset loses an edge. Most remaining "errors" on generated data are this (docs/ERROR_ANALYSIS.md).
- **One heavy job at a time:** running training or evaluation concurrently slowed both about 20× on this laptop.
- **Weights provenance:** every metrics file records the weights sha1 and config hash. Check them before comparing runs; a run once silently used smoke-test weights.
- **Refit confidence:** when graph features or thresholds change, refit the confidence model, or the scores drift.
- **Prediction files:** `<pred_dir>/<subset>/<id>.graphml`, with the drawing uid formatted as `<subset>/<id>`.
- **Docker:** CPU-only. It mounts `./weights` read-only and keeps job data in the `pid-data` volume.
