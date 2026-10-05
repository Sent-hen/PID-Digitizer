# Uses the project venv directly when present (Windows: .venv/Scripts, POSIX: .venv/bin),
# otherwise `uv run`.
ifneq ($(wildcard .venv/Scripts/python.exe),)
  PY := .venv/Scripts/python.exe
  BIN := .venv/Scripts/
else ifneq ($(wildcard .venv/bin/python),)
  PY := .venv/bin/python
  BIN := .venv/bin/
else
  PY := uv run python
  BIN := uv run
endif

TEST_SPLITS ?= --split dataset_pid_test --split synthetic_test --split open100_test
VAL_SPLITS ?= --split dataset_pid_val --split synthetic_val
RUN ?= m1b_baseline

.PHONY: install lint typecheck test test-dataset check inspect splits eval-oracle \
        tiles train baseline baseline-val baseline-oracle eval demo

install:
	uv sync

lint:
	$(BIN)ruff check src tests scripts

typecheck:
	$(BIN)mypy

test:
	$(PY) -m pytest

test-dataset:  ## needs the full dataset on disk
	$(PY) -m pytest -m dataset

check: lint typecheck test

inspect:
	$(PY) scripts/inspect_dataset.py

splits:
	$(PY) scripts/make_splits.py

eval-oracle:  ## harness sanity check: GT vs GT must score 1.0
	$(PY) scripts/evaluate.py --oracle $(TEST_SPLITS)

tiles:
	$(PY) -c "from pid.config import load_config as L, resolve as R; from pid.detect.tiles import build_cache as B; c=L('detect'); B(c['tiles']['train_splits'], R(L('data')['splits']['dir']), R(c['tiles']['cache_dir'])/'train', c, L('pipeline')['preprocess']['target_long_side'])"

train:
	$(PY) scripts/train_detector.py

baseline-val:
	$(PY) scripts/run_baseline.py --run $(RUN) $(VAL_SPLITS)

baseline:  ## Milestone 1: end-to-end on the test splits
	$(PY) scripts/run_baseline.py --run $(RUN) $(TEST_SPLITS)

baseline-oracle:  ## same pipeline with ground-truth symbols (error propagation)
	$(PY) scripts/run_baseline.py --run $(RUN)_oracle_symbols --oracle-symbols $(VAL_SPLITS) $(TEST_SPLITS)

eval: baseline

demo:
	@echo "Milestone 6: not implemented yet" && exit 1
