"""Checks against the real dataset. Skipped in CI (`-m dataset` to run locally)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from pid.config import resolve
from pid.eval.report import evaluate
from pid.io.pid2graph import iter_refs, load_drawing

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def refs(data_cfg: dict[str, Any]) -> list[Any]:
    if not (resolve(data_cfg["dataset_root"]) / data_cfg["complete_dir"]).exists():
        pytest.skip("PID2Graph not on disk")
    return list(iter_refs(data_cfg))


def test_counts(refs: list[Any]) -> None:
    by = {s: sum(r.subset == s for r in refs) for s in ("dataset_pid", "open100", "synthetic")}
    assert by == {"dataset_pid": 500, "open100": 12, "synthetic": 250}


def test_oracle_on_sampled_drawings(refs: list[Any], data_cfg: dict[str, Any]) -> None:
    picks = random.Random(0).sample(refs, 5)
    pairs = [(d, d.model_copy(deep=True)) for d in (load_drawing(r, data_cfg) for r in picks)]
    r = evaluate(pairs, data_cfg)
    assert r["edges"]["f1"] == 1.0
    assert r["node_AP50"] == pytest.approx(1.0)
    assert r["symbols"]["mAP50"] == pytest.approx(1.0)
    assert r["pid2graph_edge_map"]["edge_mAP"] == pytest.approx(1.0)


def test_splits_disjoint_and_complete(data_cfg: dict[str, Any], refs: list[Any]) -> None:
    d = resolve(data_cfg["splits"]["dir"])
    seen: list[str] = []
    for f in sorted(d.glob("*_*.txt")):
        seen += f.read_text().split()
    assert len(seen) == len(set(seen)) == len(refs)
