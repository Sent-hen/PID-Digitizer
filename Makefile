PY ?= uv run python
SPLITS ?= --split dataset_pid_test --split synthetic_test --split open100_test

.PHONY: install lint typecheck test test-dataset check inspect splits eval-oracle eval baseline demo

install:
	uv sync

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

test-dataset:  ## needs the full dataset on disk
	uv run pytest -m dataset

check: lint typecheck test

inspect:
	$(PY) scripts/inspect_dataset.py

splits:
	$(PY) scripts/make_splits.py

eval-oracle:  ## harness sanity check: GT vs GT must score 1.0
	$(PY) scripts/evaluate.py --oracle $(SPLITS)

eval:
	@echo "No model yet (Milestone 1). Running the oracle sanity check instead."
	$(MAKE) eval-oracle

baseline:
	@echo "Milestone 1: not implemented yet" && exit 1

demo:
	@echo "Milestone 6: not implemented yet" && exit 1
