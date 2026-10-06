# P&ID Digitizer: Project Proposal and Execution Plan (v3)

> v3 (2026-10-04) re-scopes v2 to what the PID2Graph dataset can train and measure, makes GraphML the primary output, and removes items that served presentation rather than function. Changes from v2 are listed in §9. Work milestone by milestone; don't start a milestone until the previous one's acceptance criteria pass.

## 1. Goal

Convert raster P&ID drawings (scans and rendered sheets) into a **graph** and export it as **GraphML** (primary, lossless) and **DEXPI-structured XML** (secondary, for plant-engineering tools).
- **Nodes:** symbols, plus the structural crossing and ankle nodes.
- **Edges:** pipes and signal lines.

The system is served through a FastAPI service with a minimal review UI, in one Docker container. It's evaluated on PID2Graph under the official protocol and our own connectivity metrics, with a robustness study and an error analysis.

Priorities, in order:
1. Honest, reproducible results (no inflated claims).
2. A working demo that someone can run in under 5 minutes.
3. Clear engineering: tests, config, typed code, a short README.

Strategy: a walking skeleton first (complete pipeline → metric in Milestone 1), then improve each stage against the metric.

## 2. Dataset facts that set the scope

Details are in DATASET_NOTES.md. These facts decide what's in and out of scope:

| Fact | Consequence |
|---|---|
| **Raster only.** 762 PNG/JPG sheets in three subsets: Dataset-PID 500 and Synthetic 250 (both generated), OPEN100 12 (real). | Raster pipeline only. The vector-PDF path is a stretch goal (§8). |
| **No training split.** All three subsets are the paper's test sets. | We make our own splits. Our numbers are reported under the official metric definitions but are not directly comparable to the paper's. |
| **Labels:** 7 symbol classes and 2 line types (solid, non-solid). Structural nodes (`crossing`, `ankle`) are 8×8 boxes. | The pipeline must output structural nodes. Line style is two-class. |
| **Crossings are always connections.** Every line intersection is one connected `crossing` node. | No junction-vs-crossing logic or metric. Graph assembly follows this convention. |
| **Implicit ports.** No port annotations, but `ankle` nodes sit exactly on symbol borders. | Ankle placement is the port model. No separate port heuristics. |
| **Off-page connectors are a class** (`inlet_outlet`). | No special node type. |
| **No text ground truth** and no tag attribute. Synthetic text is random strings. | Text is detected only to mask it before line extraction. Tag reading is a stretch goal. |
| **Ignore regions:** frame and title-block rectangles annotated in Dataset-PID and OPEN100. | A border-cropping step can be trained and measured against them. |
| **Clean digital renders.** | Robustness to real scans can only be shown with a degradation simulator. |
| **Rare classes:** tank, pump and inlet/outlet appear mostly or only in OPEN100. | Option B: 4 OPEN100 sheets go into training (oversampled about 3×), 8 stay in test. |

## 3. Ground rules

**Continuity**
- Keep `PROGRESS.md` current: milestone, done, open issues, next step. Update it every session.
- Commit after each completed task. Never leave the repo broken.

**Logs**
- `DECISIONS.md`: non-obvious choices, with the rejected alternative.
- `RESULTS.md`: every experiment with config, commit, hardware and metrics. Never report a number that isn't there.
- `DATASET_NOTES.md`: format, license, attribution, quirks.

**Engineering**
- Python 3.11+, `uv`, `ruff`, `mypy --strict`, `pytest`.
- Paths and hyperparameters go in `configs/*.yaml`. Seeds are fixed. Split ID lists are committed; data is not.
- CI is CPU only: tiny fixtures in `tests/data/`, no dataset, no GPU. Mark dataset and slow tests.
- Never commit the dataset, weights over 50 MB, caches or generated outputs. Experiment logs are JSONL written by the scripts.

**Licensing**
- Permissive licenses only; no AGPL (no Ultralytics). Record each model library's license in DECISIONS.md.
- The dataset is CC BY-SA 4.0. Derived data and published weights carry attribution and share-alike.

**Ask first** before:
- spending more than ~3 GPU-hours on one experiment;
- adding a heavy dependency or service;
- changing the GraphML schema after Milestone 4;
- any change to scope.

## 4. Output format

**GraphML (primary).** One file per drawing, readable by NetworkX, yEd, Gephi and igraph. It uses the same attribute names as the PID2Graph ground truth, so evaluation reads predictions and GT with one loader.

| Element | Attribute (`attr.name`) | Type | Notes |
|---|---|---|---|
| graph | `schema_version`, `source_file`, `width`, `height`, `pipeline_version`, `config_hash` | string/int | |
| node | `label` | string | One of the 7 symbol classes, or `crossing` / `ankle`. Raw dataset spellings, e.g. `inlet/outlet`, `connector` for ankle. |
| node | `xmin`, `ymin`, `xmax`, `ymax` | double | Original image pixels. |
| node | `confidence` | double | Calibrated from Milestone 3; detector score before that. |
| edge | `edge_label` | string | `solid` / `non-solid`. |
| edge | `confidence` | double | |

- **Validation:** against the official GraphML XSD.
- **Round trip:** GraphML → `Drawing` → GraphML must be identical.
- **Transport:** JSON exists only as the API/UI transport (the `Drawing` pydantic model). It isn't a second documented schema.

**DEXPI (secondary).** A DEXPI/Proteus-structured export that maps classes via `export/dexpi_mapping.yaml`:
- unmapped classes (e.g. `general`) are listed explicitly;
- structural nodes become piping-network connection points;
- without official schema validation plus a successful import into a DEXPI-aware tool, it's called "DEXPI-structured", never "compliant".

## 5. Repository layout

```
configs/   data.yaml pipeline.yaml detect.yaml text.yaml serve.yaml
src/pid/
  io/          # Drawing model, PID2Graph loader, overlays
  preprocess/  # rescaling, tiling, border cropping, degradation simulator
  detect/      # detector, tiled inference, tile merging
  text/        # text detection used as a mask for line extraction
  lines/       # line extraction, line-style classification
  graph/       # assembly (dataset conventions), confidence
  export/      # graphml.py, dexpi.py (+ dexpi_mapping.yaml)
  eval/        # matching, detection, connectivity, robustness
  api/ ui/     # FastAPI service, static review UI
scripts/  tests/  docker/  docs/
```

## 6. Milestones

### Milestone 0: Dataset audit, scaffolding, evaluation harness (done)
- **Audit:** formats, subsets, labels, anomalies and sample overlays (`docs/dataset_stats.md`, `docs/dataset_samples/`). DATASET_NOTES.md is complete, including the official protocol and published baselines.
- **Model and loader:** canonical `Drawing` model and PID2Graph loader, with the round trip checked visually.
- **Splits:** grouped by near-duplicate (ink-map Jaccard; no duplicates found).
- **Harness:**
  - symbol mAP@0.5 and @0.5:0.95 (symbols only, pycocotools);
  - class-agnostic node AP@0.5;
  - edge P/R/F1 with line-type accuracy and per-drawing median/p10;
  - the official PID2Graph edge mAP.
  - Oracle and perturbation tests pass.
- **Scaffolding:** CI and hardware recorded.

### Milestone 1: Walking skeleton (redo with v3 scope)
- **Splits (option B):** 4 OPEN100 sheets move to training, chosen by a seeded rare-class-aware rule so that the 8 test sheets still contain every class. Dataset-PID and Synthetic splits are unchanged.
- **Input:** rescale to a 7000 px long side, 1024² tiles with 256 overlap, coordinate mapping.
- **Detector:** torchvision Faster R-CNN (BSD-3), briefly fine-tuned on train tiles, with OPEN100 tiles oversampled about 3×.
- **Merging:** class-aware tile merging (complete-first, IoU/IoS), unit-tested at tile seams.
- **Lines:** orthogonal line extraction plus graph assembly following the dataset conventions (crossings at line meetings, ankles on symbol borders).
- **Export:** GraphML (§4 keys; confidence = detector score, 1.0 for structural nodes and edges).
- **Runs:**
  - `make baseline` on val and test.
  - `make baseline-oracle`, the same pipeline with GT symbols, for error propagation.

**Acceptance**
- `make baseline` runs on all test splits with zero crashes. Failures are caught, logged and counted.
- Every output GraphML loads with the ground-truth loader and passes the GraphML XSD.
- Baseline and oracle numbers are in RESULTS.md. These are the numbers to beat.

### Milestone 2: Detection and preprocessing
- **`preprocess/degrade.py`:** deterministic degradation (skew, noise, blur, JPEG, contrast loss, fax-like binarization) at 3 severity levels.
- **`preprocess/crop.py`:** frame and title-block detection, evaluated against the annotated ignore regions (IoU), with the effect on edge precision measured on val.
- **Detection improvements**, measured on val only:
  - class-imbalance handling;
  - augmentations, including degradation and semantics-safe flips/rotations (directional symbols noted);
  - a tile size/overlap sweep within the GPU budget.

**Acceptance**
- The degradation simulator is deterministic for a fixed seed.
- Cropping IoU is reported.
- Val symbol mAP and node AP beat Milestone 1, with a per-class AP table in RESULTS.md.

### Milestone 3: Lines, text mask, connectivity, confidence (the core)
- **Text mask:** text detection only, using a permissively licensed detector; license recorded. Masked text must not create line segments. Effect measured on val edge precision.
- **Line style:** fix dashed-line recall and classification. Milestone 1 non-solid edge AP on Dataset-PID is 0.43, against 0.93 for solid.
- **OPEN100 gaps:** diagonal lines, direct symbol–symbol edges, faint signal lines.
- **Segment merging:** collinear fragments and gap bridging.
- **`graph/confidence.py`:**
  - per-node and per-edge confidence from detector score, snap distance, segment continuity and fill ratio, with the formula documented;
  - calibrated on val (logistic regression), with a reliability diagram in `docs/`;
  - non-constant scores also make AP meaningful.
- **Fixtures:** T, X, bend, inline valve, dashed line, diagonal line, text next to a line.

**Acceptance**
- Val edge F1 and edge mAP beat Milestone 1.
- Fixture tests pass.
- The reliability diagram is published.

### Milestone 4: Export
- **GraphML:**
  - versioned schema, as in §4;
  - keys documented in `docs/GRAPHML_SCHEMA.md`;
  - XSD validation and the round-trip test.
- **DEXPI-structured XML:**
  - record the specification and schema version used;
  - mapping YAML with an explicit unmapped list;
  - generated with `lxml`;
  - validated against the XSD where available, otherwise a required-element test;
  - supported vs. unsupported features documented in DECISIONS.md.

**Acceptance**
- GraphML round trip is identical and XSD-valid.
- The DEXPI validation (or required-element) test passes.
- The mapping coverage table is published.

### Milestone 5: Evaluation and error analysis
- **Final metrics** on test, run once per final config:
  - the official protocol (symbol mAP, node AP, edge mAP);
  - our edge P/R/F1 with per-drawing median/p10;
  - line-type accuracy.
- **Robustness:** the full pipeline at 3 degradation severities, showing the per-stage metric drop. This is where preprocessing toggles are ablated, because they have no effect on clean renders.
- **Ablations**, only those tied to design decisions: tile overlap, border cropping, text mask, confidence calibration, OPEN100 oversampling.
- **Error propagation:** oracle symbols vs. predicted symbols.
- **`docs/ERROR_ANALYSIS.md`:**
  - error categories: missed symbols, class confusion, false positives, broken pipes, spurious connections, dashed-line errors, text-induced segments;
  - each with counts, 3–5 examples, probable cause and fixes tried.
- **Limitations:**
  - generated vs. real domain gap;
  - coarse 7-class taxonomy;
  - OPEN100 test size (8 sheets);
  - crossing convention;
  - DEXPI coverage.

**Acceptance**
- `make eval` regenerates every table and figure.
- RESULTS.md's final table matches the README.

### Milestone 6: Service, review UI, container
- **API:**
  - endpoints: `POST /jobs` (upload), `GET /jobs/{id}`, `GET /jobs/{id}/result?format=graphml|dexpi`, `GET /jobs/{id}/overlay`, `PATCH /jobs/{id}/graph` (corrections), `GET /health`;
  - an in-process worker with SQLite job state;
  - upload size/type limits, a timeout, structured logs;
  - models loaded once.
- **Review UI** (static):
  - pan/zoom with an overlay colored by confidence, plus a low-confidence filter;
  - click to inspect;
  - relabel or delete nodes and edges, and add an edge between two nodes;
  - download GraphML/DEXPI.
  - Corrections are stored as an append-only log next to the corrected graph.
- **Container:** CPU Dockerfile and `docker-compose.yml`, with a healthcheck, a non-root user and pinned versions. Weights are fetched by a script. `make demo` starts it with a bundled sample drawing.

**Acceptance**
- From a clean clone, `docker compose up` processes the sample drawing in the UI.
- API tests (pytest + httpx) cover upload, polling, GraphML/DEXPI retrieval, and a correction round trip.

## 7. README
Contents:
1. A one-paragraph pitch.
2. The results table, naming the dataset, splits and baselines.
3. Limitations.
4. Quickstart (Docker) and developer setup.
5. Exact retrain and re-evaluate commands.
6. License and dataset attribution.

## 8. Stretch (only after the minimum cut is done)
- **Vector-PDF path:** page classification and PyMuPDF line/text extraction. Validated only on 3–5 self-made vector PDFs in `tests/data/`, with that scope stated in the README.
- **Tag reading:** OCR plus tag grammar plus association, reported only on a hand-labelled sample with its size stated.

## 9. Changes from v2

| v2 item | v3 | Reason |
|---|---|---|
| JSON primary + DEXPI | GraphML primary + DEXPI. JSON only as API transport. | GraphML is the dataset's GT format (one loader, exact round trip). Two lossless schemas would duplicate each other. |
| Vector PDF path, PDF classifier, vector ablation | Stretch | No PDFs in the dataset |
| OCR, tag grammar, association, OCR metrics | Text-detection mask only; tags are stretch | No text GT, and no metric uses text |
| Junction-vs-crossing logic and metric | Removed | The GT connects every intersection |
| Port heuristics and port-snapping ablation | Removed | Ankle nodes on symbol borders already encode ports |
| Off-page connector node type | Removed | It's the `inlet_outlet` class |
| Edge polylines | Removed | Not in the GT; derivable from node positions |
| Cross-subset test (synthetic → real) | Replaced by the held-out OPEN100 test (option B: 8 sheets) | Subset class sets barely overlap |
| Clean-image preprocessing ablations | Folded into the robustness study | No effect on clean renders |
| Learned image-to-graph comparison | Cite the paper's numbers only (done) | Code and training data aren't released |
| Demo GIF, architecture diagram, ONNX comparison, per-stage benchmark, Redis option, UI undo/redo and diff view, active-learning note | Removed | Presentation only; no functional or evaluative need |
| (missing) | Added: structural nodes, scale normalization, border cropping from ignore regions, dashed/diagonal-line work, option B | Required by the dataset and the Milestone 1 findings |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Generated data is far cleaner than real scans | Degradation robustness study, the OPEN100 real test, an explicit Limitations section |
| OPEN100 test is only 8 sheets | Report per-drawing distributions and per-class counts; no claims beyond them |
| Data leakage | Grouped splits; tile-index leakage check; test run once per final config |
| Detection errors hide graph-logic quality | Oracle-symbol error propagation |
| DEXPI mapping incomplete | Unmapped-class list; no "compliant" claim without validation |
| Limited GPU (8 GB laptop) | Cached tiles, AMP, ≤3 GPU-hours per experiment |

## 11. Timeline and minimum cut

| Milestone | Weeks (part time) |
|---|---|
| 0 Audit, harness | done |
| 1 Walking skeleton (redo) | 0.5 |
| 2 Detection + preprocessing | 1.5 |
| 3 Lines, text mask, connectivity, confidence | 2.5 |
| 4 Export | 1 |
| 5 Evaluation | 1 |
| 6 Service, UI, container | 1.5 |

**Minimum cut:** Milestones 0–3, GraphML export, Milestone 5's metrics with error propagation and error analysis, and a read-only UI. DEXPI and UI editing come after.
