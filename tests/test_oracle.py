"""Oracle tests for the evaluation harness: ground truth scores 1.0, and known
perturbations reduce scores by exactly the expected amount."""

from __future__ import annotations

from typing import Any

import pytest

from pid.eval.connectivity import edge_map_pid2graph, edge_prf
from pid.eval.detection import node_ap, symbol_map
from pid.eval.report import evaluate
from pid.io.drawing import BBox, Drawing, Edge, LineType

N_EDGES = 60  # 6x6 grid: 2 * 6 * 5


def test_grid_fixture_shape(grid: Drawing) -> None:
    assert len(grid.symbols) == 36 and len(grid.edges) == N_EDGES
    assert sum(e.line_type == LineType.NON_SOLID for e in grid.edges) == 20


def test_ground_truth_scores_perfect(grid: Drawing, mini: Drawing,
                                     data_cfg: dict[str, Any]) -> None:
    for d in (grid, mini):
        r = evaluate([(d, d.model_copy(deep=True))], data_cfg)
        assert r["edges"]["f1"] == 1.0
        assert r["edges"]["line_type_accuracy"] == 1.0
        assert r["node_AP50"] == pytest.approx(1.0)
        assert r["symbols"]["mAP50"] == pytest.approx(1.0)
        assert r["symbols"]["mAP50_95"] == pytest.approx(1.0)
        assert r["pid2graph_edge_map"]["edge_mAP"] == pytest.approx(1.0)


@pytest.mark.parametrize("k", [1, 6, 30])
def test_deleting_edges_reduces_recall_exactly(grid: Drawing, k: int) -> None:
    pred = grid.model_copy(deep=True)
    pred.edges = pred.edges[k:]
    c = edge_prf(grid, pred)
    assert (c.tp, c.fp, c.fn) == (N_EDGES - k, 0, k)
    assert c.precision == 1.0
    assert c.recall == pytest.approx((N_EDGES - k) / N_EDGES)


def test_spurious_edges_reduce_precision_exactly(grid: Drawing) -> None:
    pred = grid.model_copy(deep=True)
    # diagonal neighbors are never GT edges
    extra = [Edge(id=f"x{i}", source=f"n{i}_{i}", target=f"n{i + 1}_{i + 1}") for i in range(5)]
    pred.edges = pred.edges + extra
    c = edge_prf(grid, pred)
    assert (c.tp, c.fp, c.fn) == (N_EDGES, 5, 0)
    assert c.precision == pytest.approx(N_EDGES / (N_EDGES + 5))


def test_duplicate_predicted_edge_is_fp(grid: Drawing) -> None:
    pred = grid.model_copy(deep=True)
    e = pred.edges[0]
    pred.edges.append(Edge(id="dup", source=e.target, target=e.source, line_type=e.line_type))
    c = edge_prf(grid, pred)
    assert (c.tp, c.fp, c.fn) == (N_EDGES, 1, 0)


def test_missing_node_loses_its_edges(grid: Drawing) -> None:
    pred = grid.model_copy(deep=True)
    victim = "n2_2"  # interior node: degree 4
    pred.symbols = [s for s in pred.symbols if s.id != victim]
    pred.edges = [e for e in pred.edges if victim not in (e.source, e.target)]
    c = edge_prf(grid, pred)
    assert (c.tp, c.fp, c.fn) == (N_EDGES - 4, 0, 4)


def test_box_shift_below_iou_unmatches_node(grid: Drawing) -> None:
    pred = grid.model_copy(deep=True)
    s = next(s for s in pred.symbols if s.id == "n0_0")  # corner node: degree 2
    b = s.bbox
    s.bbox = BBox(xmin=b.xmin + 20, ymin=b.ymin, xmax=b.xmax + 20, ymax=b.ymax)  # IoU 10/50
    c = edge_prf(grid, pred)
    assert (c.tp, c.fp, c.fn) == (N_EDGES - 2, 2, 2)


def test_line_type_flip(grid: Drawing) -> None:
    pred = grid.model_copy(deep=True)
    for e in pred.edges[:6]:
        e.line_type = LineType.SOLID if e.line_type == LineType.NON_SOLID else LineType.NON_SOLID
    c = edge_prf(grid, pred)
    assert c.f1 == 1.0  # topology unaffected
    assert c.line_type_accuracy == pytest.approx((N_EDGES - 6) / N_EDGES)
    # Official edge mAP counts a wrong class as FP and (not FN): AP drops below 1.
    assert edge_map_pid2graph([(grid, pred)])["edge_mAP"] < 1.0


def test_edge_map_low_score_fp_ranked_last(grid: Drawing) -> None:
    pred = grid.model_copy(deep=True)
    pred.edges.append(Edge(id="x", source="n0_0", target="n1_1", line_type=LineType.SOLID,
                           score=0.01))
    # an FP ranked below all TPs does not reduce all-point AP
    assert edge_map_pid2graph([(grid, pred)])["AP_solid"] == pytest.approx(1.0)


def test_node_ap_ignores_class_symbol_map_does_not(grid: Drawing,
                                                   data_cfg: dict[str, Any]) -> None:
    pred = grid.model_copy(deep=True)
    for s in pred.symbols:
        if s.cls == "valve":
            s.cls = "general"
    all_cls = data_cfg["symbol_classes"] + data_cfg["structural_classes"]
    assert node_ap([(grid, pred)], all_cls) == pytest.approx(1.0)
    assert symbol_map([(grid, pred)], data_cfg["symbol_classes"])["mAP50"] < 1.0


def test_dropped_symbols_reduce_symbol_recall(grid: Drawing, data_cfg: dict[str, Any]) -> None:
    pred = grid.model_copy(deep=True)
    valves = [s.id for s in pred.symbols if s.cls == "valve"]
    drop = set(valves[: len(valves) // 2])
    pred.symbols = [s for s in pred.symbols if s.id not in drop]
    pred.edges = [e for e in pred.edges if not ({e.source, e.target} & drop)]
    r = symbol_map([(grid, pred)], data_cfg["symbol_classes"])
    assert r["per_class_AP50"]["general"] == pytest.approx(1.0)
    # pycocotools samples recall at 101 points; half the valves -> AP ~= 0.5
    assert r["per_class_AP50"]["valve"] == pytest.approx(0.5, abs=0.01)
