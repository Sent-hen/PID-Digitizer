# P&ID Digitizer: Project Proposal and Execution Plan (v2)

> Hand this file to Claude Code as the project spec (save it as `docs/PROPOSAL.md` and reference it from `CLAUDE.md`). Work milestone by milestone. Do not start a milestone until the previous one's acceptance criteria pass.

## 1. Goal

Build an end-to-end system that converts legacy P&ID drawings (raster scans, rasterized PDFs, and vector PDFs) into a **graph** (equipment, valves, instruments as nodes; pipes and signal lines as edges) and exports it as **JSON** and **DEXPI/Proteus-structured XML**. It is wrapped in a **FastAPI service with a review UI**, containerized with Docker, and evaluated with **graph-level metrics**, a **robustness study**, and an **error analysis**.

This is a portfolio project. Priorities, in order:
1. Honest, reproducible results (no inflated claims).
2. A working demo that someone can run in under 5 minutes.
3. Clear engineering: tests, config, typed code, clean README.

### Strategy: walking skeleton first

Build a crude but complete pipeline (image -> graph -> JSON -> metric) in Milestone 1, then improve each stage against the metric. This surfaces integration problems early (especially connectivity, the hardest part) and guarantees a demoable result at every point, instead of discovering in week 8 that the stages don't fit together.

## 2. Ground rules for Claude Code

**Session continuity**
- Maintain `PROGRESS.md`: current milestone, completed tasks, open issues, and the next concrete step. Update it at the end of every work session so a fresh session can resume without re-deriving context.
- Commit after each completed task with a descriptive message. Never leave the repo in a broken state at the end of a session.

**Logs**
- `DECISIONS.md`: every non-obvious choice, with the reason and the alternative rejected.
- `RESULTS.md`: every experiment with config file, commit hash, hardware, and metrics. Never report a number that is not in this file.
- `DATASET_NOTES.md`: dataset format, license, attribution, and quirks.

**Engineering**
- Python 3.11+, `uv` for pinned dependencies, `ruff` + `mypy` + `pytest`.
- All paths and hyperparameters live in YAML configs (`configs/`), not in code.
- Fix random seeds. Commit split ID lists (not data).
- CI runs on CPU only: tests must use tiny fixtures under `tests/data/` and never require the full dataset or a GPU. Mark slow/integration tests separately.
- Do **not** commit the dataset, model weights over ~50 MB, cached tiles, or generated outputs.
- Lightweight experiment tracking: a CSV/JSONL log written by training and eval scripts (MLflow is optional; don't add infrastructure for its own sake).

**Licensing**
- Check licenses for every model library before adopting it, and record them in `DECISIONS.md`. In particular, Ultralytics YOLO is AGPL-3.0, which affects how the repo and any service built on it can be licensed. Prefer an Apache/MIT-licensed detector (e.g. torchvision detection models, YOLOX, RT-DETR implementations under permissive licenses, or MMDetection) unless there is a documented reason not to.
- Check and record the dataset's license before publishing weights or derived data.

**When to ask me first**
- Spending more than ~3 GPU-hours on a single experiment.
- Adding a heavy dependency or a new service (Redis, a DB server, etc.).
- Changing the output schema after Milestone 4.
- Any change to this plan's scope.

## 3. Repository layout

```
pid-digitizer/
  README.md  CLAUDE.md  PROGRESS.md  DECISIONS.md  RESULTS.md  DATASET_NOTES.md
  pyproject.toml  Makefile
  configs/   data.yaml detect.yaml ocr.yaml lines.yaml graph.yaml serve.yaml
  src/pid/
    io/          # loaders, canonical Drawing model, PDF classification, vector extraction
    preprocess/  # deskew, denoise, binarize, tiling, degradation simulator
    detect/      # symbol detection, tiled inference, box merging
    ocr/         # text detection/recognition, tag parsing, association
    lines/       # masking, line detection, line-style classification, segment merging
    graph/       # schema, port inference, assembly, confidence scoring
    export/      # json_export.py, dexpi_export.py, validation
    eval/        # detection, OCR, connectivity metrics, robustness, error analysis
    api/         # FastAPI app, job runner, persistence
    ui/          # review UI (static, served by FastAPI)
  scripts/       # inspect, convert, train, evaluate, benchmark
  tests/         # unit + small integration fixtures
  docker/
  docs/          # architecture diagram, figures, demo GIF, JSON Schema, error analysis
```

## 4. Milestones

### Milestone 0: Dataset audit, scaffolding, and evaluation harness

**Dataset audit**
- Inspect the PID2Graph directory: file formats, image sizes, annotation format (boxes, graph files, class lists, ports if present), subsets (e.g. drawings derived from real plants vs. synthetic), and any official splits.
- `scripts/inspect_dataset.py`: counts per subset, image size distribution, class frequency table, nodes/edges per drawing, line-type distribution, and sample overlays saved to `docs/dataset_samples/`.
- Read the dataset's accompanying paper/README. Record any **official evaluation protocol and published baseline numbers** in `DATASET_NOTES.md`. If they exist, our connectivity metric must also be reported under that protocol so results are comparable.
- Convert annotations to a canonical pydantic `Drawing` model (`symbols`, `texts`, `edges`, `ports`).

**Splits (leakage control)**
- Use official splits if provided. Otherwise split by drawing, and additionally **group by source/template** where drawings share a generator template, sheet series, or near-duplicate layout (detect near-duplicates with perceptual hashing). Report both an in-distribution test set and, if the dataset has distinct subsets, a cross-subset test (e.g. train on synthetic, test on real-derived).

**Evaluation harness (built now, not at the end)**
- `eval/match.py`: IoU-based matching of predicted to ground-truth symbols.
- `eval/connectivity.py`: edge precision/recall/F1 where an edge is correct only if both endpoints match the correct ground-truth nodes; line-type accuracy reported separately.
- `eval/detection.py`: mAP@0.5, mAP@0.5:0.95, per-class AP (use an established implementation such as `pycocotools` or `torchmetrics`; don't hand-roll mAP).
- **Oracle tests:** feeding ground truth through the harness must score 1.0; deleting/perturbing a known number of edges must reduce scores by the expected amount.

**Scaffolding**
- Repo layout, CI (lint, type check, CPU tests), Makefile targets (`make test`, `make eval`, `make demo`), and configs.
- Detect hardware (`nvidia-smi`, RAM, disk) and record it in `DECISIONS.md`. Estimate disk needs for tile caches before generating them.

**Acceptance criteria**
- `inspect_dataset.py` produces stats and overlays; `DATASET_NOTES.md` complete.
- Annotation round-trip (dataset -> canonical model -> overlay) visually correct on 5 sampled drawings.
- Harness oracle tests pass.

---

### Milestone 1: Walking skeleton (end-to-end baseline)

Goal: the simplest pipeline that produces a scored graph for every test drawing.

- Raster input only. Fixed-size overlapping tiling with coordinate mapping (`preprocess/tiling.py`).
- Detector: an off-the-shelf permissively licensed model fine-tuned briefly (short schedule is fine) on tiles.
- Tile merging: class-aware NMS in `detect/merge.py` (own implementation, unit-tested on objects spanning tile borders).
- Lines: mask detected symbols, binarize, skeletonize, extract segments, snap endpoints to the nearest symbol bounding box within a tolerance.
- No OCR yet; no confidence scores yet.
- Export: minimal JSON (nodes, edges).
- Run the harness on validation and test; log baseline numbers in `RESULTS.md`.

**Acceptance criteria**
- `make baseline` runs end to end on the test split with zero crashes (failures caught, logged, counted).
- Baseline detection and connectivity numbers recorded. These are the numbers every later milestone must beat.

---

### Milestone 2: Ingestion paths and detection improvements

**Ingestion**
- `io/classify.py`: per-page vector vs. raster classification for PDFs, with confidence and reason.
- **Vector path** (PyMuPDF): extract segments, curves, rectangles, and text spans with position, rotation, and size into the same intermediate representation used by the raster path, so downstream stages skip OCR/line detection when vector data exists. Symbols still need detection (render the page and run the detector), but lines and text come from the vector data.
- **Vector test data:** PID2Graph is likely raster-only. Validate the vector path on a small set (3-5) of public-domain or self-made vector P&ID PDFs in `tests/data/`. State exactly this scope in the README. Do not attempt large-scale vector evaluation.
- Raster preprocessing: optional deskew, denoise, adaptive binarization, title-block/border cropping, each toggleable in config and ablated.
- `preprocess/degrade.py`: reproducible degradation simulator (skew, noise, blur, JPEG artifacts, contrast loss, fax-like binarization) at 3 severity levels.

**Detection**
- Tune tile size/overlap, input resolution, and model size against the hardware budget.
- Class imbalance strategy (oversampling, class weighting, targeted augmentation), documented.
- Augmentations that respect symbol semantics (no mirroring of directional symbols without a note). Include degradation-style augmentations for robustness.
- Optional comparison: weighted box fusion vs. NMS for tile merging.

**Acceptance criteria**
- PDF classifier unit tests on at least 4 files; tiling round-trip within 1 px; degradation simulator deterministic for a fixed seed.
- Detection beats the Milestone 1 baseline; per-class AP table in `RESULTS.md`.

---

### Milestone 3: Text, lines, and connectivity (the core)

Budget the most time here.

**Text and tags**
- OCR with an open engine (e.g. PaddleOCR); compare against Tesseract on validation crops and record the choice.
- Handle rotated text (at least 0/90/180/270).
- `ocr/tags.py`: configurable regex/grammar for instrument tags, equipment tags, and line numbers; at least 30 unit-test strings including malformed ones.
- `ocr/associate.py`: associate text to symbols/lines using distance and orientation, with an association confidence.
- If the dataset lacks text ground truth, hand-label a small set (~20 drawings or ~300 text crops) with a simple labeling script, and report the sample size next to every OCR number.

**Lines and graph**
- Line-style classification (solid process vs. dashed/dotted signal, plus any other dataset types), rules-based first.
- Segment merging: collinear fragments, small-gap bridging, near-orthogonal snapping.
- **Junction vs. crossing:** explicit logic (junction-dot detection, T vs. X intersections), evaluated as its own metric.
- `graph/ports.py`: use dataset port annotations if available; otherwise class-specific port heuristics. Snap line endpoints to ports.
- `graph/assemble.py`: NetworkX graph with explicit node types for junctions, inline components (valves on a pipe), and off-page connectors.
- Tiny synthetic fixtures for unit tests: T-junction, crossing without junction, inline valve, off-page connector, dashed signal line.

**Confidence**
- `graph/confidence.py`: per-node and per-edge confidence from detector score, snap distance, segment continuity, and junction ambiguity. Document the formula.
- Calibrate on validation (e.g. logistic regression over those features or temperature scaling) and save a reliability diagram to `docs/`. Confidence drives the review UI, so it must be meaningful.

**Stretch ablation (only if time allows):** compare the modular pipeline with a learned image-to-graph approach if the dataset paper reports or releases one. Reproduce or cite its numbers; don't implement a new end-to-end model from scratch.

**Acceptance criteria**
- Connectivity F1 beats the baseline; junction/crossing accuracy reported.
- All fixture tests pass; a visualization utility draws the predicted graph colored by confidence.

---

### Milestone 4: Export (JSON, then DEXPI/Proteus XML)

**JSON**
- Versioned schema (`schema_version`) as pydantic models; publish the JSON Schema to `docs/`.
- Contents: metadata (source file, page, pipeline version, config hash), nodes (id, class, tag, bbox, ports, confidence), edges (id, source, target, line type, polyline, confidence), texts with associations.
- Round-trip loader test (JSON -> graph -> JSON is identical).

**DEXPI / Proteus XML**
- Obtain the current DEXPI specification and Proteus XML schema (fetch if network allows, otherwise ask me). Record the version.
- `export/dexpi_mapping.yaml`: dataset classes -> DEXPI classes. List unmapped classes explicitly; don't guess.
- Generate with `lxml`; validate against the XSD where available.
- Claim wording: call it "DEXPI-structured export" unless it passes official schema validation and opens correctly in at least one DEXPI-aware tool. Document supported vs. unsupported features in `DECISIONS.md`.

**Acceptance criteria**
- JSON round-trip test passes; JSON Schema published.
- XML validation test (or required-element test if no XSD), and a mapping coverage table.

---

### Milestone 5: Evaluation report and error analysis

The harness already exists; this milestone produces the final, defensible results.

- **Final metrics** on the held-out test set, run once per final config (no tuning on test): detection, OCR/tags, association, connectivity (per-drawing distribution: median and p10, not only the mean), junction/crossing accuracy. Also report under the dataset's official protocol if one exists, next to published baselines.
- **Robustness:** full pipeline under the degradation simulator at 3 severity levels; plot metric drop per stage to show where degradation hurts most.
- **Ablations:** preprocessing toggles, tile overlap, port snapping on/off, confidence calibration on/off, and (where applicable) vector shortcut vs. raster path on the vector test files.
- **Error propagation:** run connectivity with ground-truth symbols substituted for predicted ones. The gap between "oracle detection" and "predicted detection" connectivity shows how much error comes from detection vs. line/graph logic. This is a strong, cheap analysis.
- **Error analysis** (`docs/ERROR_ANALYSIS.md`): categorize failures (missed symbols, class confusion, false positives, OCR errors, broken pipes, spurious connections, junction/crossing mistakes, dashed-line errors), with counts, 3-5 annotated examples each, probable causes, and fixes tried.
- **Limitations:** domain gap (dataset vs. real legacy scans), symbol standards covered, image sizes tested, vector path scope, DEXPI coverage.

**Acceptance criteria**
- `make eval` regenerates every table and figure from scratch.
- `RESULTS.md` final summary table matches the README exactly.

---

### Milestone 6: Service, review UI, and containerization

**API (FastAPI)**
- `POST /jobs` (upload + options) -> `job_id`; `GET /jobs/{id}` (status, progress); `GET /jobs/{id}/result?format=json|dexpi`; `GET /jobs/{id}/overlay`; `PATCH /jobs/{id}/graph` (apply corrections); `GET /health`. OpenAPI docs enabled.
- Job execution: an in-process worker pool with SQLite-backed job state is enough. If you add Redis + RQ/Celery to demonstrate distributed design, justify it in `DECISIONS.md` and keep the single-container mode working.
- Upload limits, content-type validation, page-count limits, timeouts, and structured logging. Load models once at startup.

**Review UI (tiered)**
- **Required:** pan/zoom viewer, overlay colored by confidence, "low-confidence only" filter, click to inspect an element, relabel/retag or delete nodes and edges, download corrected JSON/XML.
- **Stretch:** add an edge by clicking two ports, undo/redo, side-by-side before/after diff.
- Store corrections as an append-only log (natural input for future active learning; mention as future work).

**Containerization and performance**
- Multi-stage CPU `Dockerfile` (GPU variant optional), `docker-compose.yml`, healthcheck, non-root user, pinned versions. Weights fetched by a documented script.
- Export the detector to ONNX and run CPU inference with ONNX Runtime if it is meaningfully faster; record the comparison.
- `scripts/benchmark.py`: p50/p95 latency per drawing and per stage, on CPU (and GPU if available).

**Demo**
- 60-90 second GIF/video: upload -> overlay -> fix an error -> download JSON/XML, embedded at the top of the README.
- `make demo` starts the stack with a bundled sample drawing.

**Acceptance criteria**
- From a clean clone: `docker compose up`, open the UI, process the sample drawing.
- API tests (pytest + httpx) cover upload, polling, result retrieval, and a correction round-trip.

## 5. README requirements

1. One-paragraph pitch and demo GIF at the top.
2. Architecture diagram.
3. Results table (detection, OCR, connectivity, robustness), with dataset, split, and baseline comparison named.
4. "What I'd do next" and "Limitations" sections.
5. Quickstart (Docker) and developer setup.
6. Reproducibility: exact commands to retrain and re-evaluate.
7. License and dataset attribution.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stages don't integrate; connectivity problems found late | Walking skeleton in Milestone 1; harness from Milestone 0 |
| Data leakage inflates scores | Group splits by template/source; near-duplicate detection; cross-subset test |
| Dataset cleaner than real scans | Degradation simulator, robustness plots, explicit Limitations |
| Vector path untested at scale | Small vector test set; scope stated plainly in README |
| Detection errors mask graph-logic quality | Oracle-detection error-propagation analysis |
| Restrictive model license | Prefer Apache/MIT detectors; record licenses |
| Limited GPU | Smaller models, mixed precision, cached tiles; log hardware with results |
| DEXPI mapping incomplete | Unmapped-class list; no "compliant" claim without validation |
| Scope creep / limited time | Tiered UI; stretch items clearly marked; minimum cut defined below |
| Lost context between Claude Code sessions | `PROGRESS.md` updated every session; small commits |

## 7. Timeline and minimum cut

Rough part-time estimate (~10-15 hrs/week alongside coursework):

| Milestone | Weeks |
|---|---|
| 0: Audit, scaffolding, harness | 1 |
| 1: Walking skeleton | 1-1.5 |
| 2: Ingestion + detection | 1.5-2 |
| 3: Text, lines, connectivity | 2.5-3 |
| 4: Export | 1 |
| 5: Evaluation report | 1 |
| 6: Service, UI, Docker, demo | 1.5-2 |
| **Total** | **~10-12** |

**Minimum impressive cut** (if time runs short): Milestones 0, 1, and 3 (with OCR), JSON export, Milestone 5's metrics plus the oracle error-propagation analysis and error analysis, and a basic read-only demo UI. Finish DEXPI, editing, and ONNX afterward.

## 8. First instruction to Claude Code

> Read `docs/PROPOSAL.md`. Begin Milestone 0 only. Inspect the dataset at `<DATASET_PATH>` and report: file formats, subsets, annotation structure, any official splits/evaluation protocol/baselines, license, and detected hardware. Then propose concrete adjustments to this plan (with reasons) before writing pipeline code. Create `PROGRESS.md` and keep it updated.
