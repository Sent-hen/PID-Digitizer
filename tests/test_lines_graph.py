"""Synthetic fixtures for line extraction and graph assembly."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from pid.config import load_config
from pid.eval.connectivity import edge_prf
from pid.graph.assemble import assemble
from pid.io.drawing import BBox, Drawing, Edge, LineType, Symbol
from pid.lines.extract import extract_segments


@pytest.fixture(scope="module")
def pcfg() -> dict[str, Any]:
    return load_config("pipeline")


def canvas(w: int = 800, h: int = 600) -> np.ndarray:
    return np.full((h, w), 255, np.uint8)


def hline(img: np.ndarray, y: int, x0: int, x1: int, t: int = 5, dash: int = 0) -> None:
    if dash:
        for x in range(x0, x1, 2 * dash):
            cv2.rectangle(img, (x, y - t // 2), (min(x + dash, x1), y + t // 2), 0, -1)
    else:
        cv2.rectangle(img, (x0, y - t // 2), (x1, y + t // 2), 0, -1)


def vline(img: np.ndarray, x: int, y0: int, y1: int, t: int = 5) -> None:
    cv2.rectangle(img, (x - t // 2, y0), (x + t // 2, y1), 0, -1)


def sym(i: str, x0: float, y0: float, x1: float, y1: float, cls: str = "valve") -> Symbol:
    return Symbol(id=i, cls=cls, bbox=BBox(xmin=x0, ymin=y0, xmax=x1, ymax=y1))


def run(img: np.ndarray, symbols: list[Symbol], pcfg: dict[str, Any]) -> Drawing:
    segs = extract_segments(img, [s.bbox for s in symbols], pcfg["lines"])
    return assemble("t", symbols, segs, pcfg["graph"])


def degree(d: Drawing) -> dict[str, int]:
    deg = {s.id: 0 for s in d.symbols}
    for e in d.edges:
        deg[e.source] += 1
        deg[e.target] += 1
    return deg


def test_inline_valve_between_two_tanks(pcfg: dict[str, Any]) -> None:
    img = canvas()
    hline(img, 300, 100, 700)
    s = [sym("A", 60, 250, 100, 350, "tank"), sym("V", 380, 285, 420, 315),
         sym("B", 700, 250, 740, 350, "tank")]
    for b in s:
        cv2.rectangle(img, (int(b.bbox.xmin), int(b.bbox.ymin)),
                      (int(b.bbox.xmax), int(b.bbox.ymax)), 0, 2)
    d = run(img, s, pcfg)
    # A - ankle - ankle - V - ankle - ankle - B : 4 ankles, 6 edges
    kinds = sorted(x.cls for x in d.symbols)
    assert kinds.count("ankle") == 4 and kinds.count("crossing") == 0
    assert len(d.edges) == 6
    ankles = [x for x in d.symbols if x.cls == "ankle"]
    xs = sorted(round(a.bbox.center[0]) for a in ankles)
    assert xs == [100, 380, 420, 700]  # ankles sit exactly on the symbol borders
    assert all(abs(a.bbox.center[1] - 300) <= 1 for a in ankles)


def test_t_junction(pcfg: dict[str, Any]) -> None:
    img = canvas()
    hline(img, 300, 100, 700)
    vline(img, 400, 300, 500)
    s = [sym("A", 60, 280, 100, 320), sym("B", 700, 280, 740, 320),
         sym("C", 380, 500, 420, 540)]
    d = run(img, s, pcfg)
    deg = degree(d)
    crossings = [x for x in d.symbols if x.cls == "crossing"]
    assert len(crossings) == 1 and deg[crossings[0].id] == 3
    cx, cy = crossings[0].bbox.center
    assert abs(cx - 400) <= 1 and abs(cy - 300) <= 1


def test_x_crossing_is_one_degree4_node(pcfg: dict[str, Any]) -> None:
    """Dataset convention: crossing lines are connected through one node."""
    img = canvas()
    hline(img, 300, 100, 700)
    vline(img, 400, 100, 500)
    s = [sym("L", 60, 280, 100, 320), sym("R", 700, 280, 740, 320),
         sym("T", 380, 60, 420, 100), sym("Bo", 380, 500, 420, 540)]
    d = run(img, s, pcfg)
    crossings = [x for x in d.symbols if x.cls == "crossing"]
    assert len(crossings) == 1 and degree(d)[crossings[0].id] == 4


def test_bend_is_degree2_crossing(pcfg: dict[str, Any]) -> None:
    img = canvas()
    hline(img, 300, 100, 402)
    vline(img, 400, 298, 500)
    s = [sym("A", 60, 280, 100, 320), sym("B", 380, 500, 420, 540)]
    d = run(img, s, pcfg)
    crossings = [x for x in d.symbols if x.cls == "crossing"]
    assert len(crossings) == 1 and degree(d)[crossings[0].id] == 2


def test_dashed_signal_line(pcfg: dict[str, Any]) -> None:
    img = canvas()
    hline(img, 300, 100, 700, t=3, dash=12)
    s = [sym("A", 60, 280, 100, 320, "instrumentation"), sym("B", 700, 280, 740, 320)]
    d = run(img, s, pcfg)
    line_edges = [e for e in d.edges if {e.source, e.target}.isdisjoint({"A", "B"})]
    assert len(line_edges) == 1 and line_edges[0].line_type == LineType.NON_SOLID


def test_text_and_frame_are_ignored(pcfg: dict[str, Any]) -> None:
    img = canvas(1000, 800)
    hline(img, 300, 100, 700)
    cv2.rectangle(img, (2, 2), (997, 797), 0, 5)  # frame
    cv2.putText(img, "AB-04958 HHHH", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    s = [sym("A", 60, 280, 100, 320), sym("B", 700, 280, 740, 320)]
    d = run(img, s, pcfg)
    assert len(d.edges) == 3  # A-ankle, ankle-ankle, ankle-B


def test_end_to_end_matches_gt_graph(pcfg: dict[str, Any]) -> None:
    """Draw a GT-style graph, run lines+assembly with GT symbols, expect F1 = 1."""
    img = canvas()
    hline(img, 300, 100, 380)
    hline(img, 300, 420, 700)
    vline(img, 550, 300, 500)
    s = [sym("A", 60, 280, 100, 320), sym("V", 380, 285, 420, 315),
         sym("B", 700, 280, 740, 320), sym("C", 530, 500, 570, 540)]

    def node(i: str, x: float, y: float, cls: str) -> Symbol:
        return sym(i, x - 4, y - 4, x + 4, y + 4, cls)

    gt_nodes = [node("a1", 100, 300, "ankle"), node("a2", 380, 300, "ankle"),
                node("a3", 420, 300, "ankle"), node("a4", 700, 300, "ankle"),
                node("a5", 550, 500, "ankle"), node("x", 550, 300, "crossing")]
    pairs = [("A", "a1"), ("a1", "a2"), ("a2", "V"), ("V", "a3"), ("a3", "x"), ("x", "a4"),
             ("a4", "B"), ("x", "a5"), ("a5", "C")]
    gt = Drawing(id="t", symbols=s + gt_nodes,
                 edges=[Edge(id=f"e{i}", source=u, target=v) for i, (u, v) in enumerate(pairs)])
    pred = run(img, s, pcfg)
    c = edge_prf(gt, pred)
    assert (c.tp, c.fp, c.fn) == (9, 0, 0)


def test_outline_outside_box_creates_no_edges(pcfg: dict[str, Any]) -> None:
    """A rectangular symbol whose outline is drawn just outside its annotated box must
    not produce ankle-ankle edges along its own border."""
    img = canvas()
    hline(img, 300, 100, 380)
    hline(img, 300, 460, 700)
    hline(img, 255, 380, 460, t=3)  # top outline drawn 5 px above the annotated box
    s = [sym("A", 60, 280, 100, 320), sym("V", 380, 260, 460, 340),
         sym("B", 700, 280, 740, 320)]
    d = run(img, s, pcfg)
    owner_v = [e for e in d.edges if "V" in (e.source, e.target)]
    assert len(owner_v) == 2  # only the two real ports of V
    assert len(d.edges) == 6


def test_text_stroke_touching_symbol_is_not_a_port(pcfg: dict[str, Any]) -> None:
    img = canvas()
    hline(img, 300, 100, 380)
    vline(img, 400, 200, 284)          # a tall text-like stroke ending on V's top border
    s = [sym("A", 60, 280, 100, 320), sym("V", 380, 285, 420, 315)]
    d = run(img, s, pcfg)
    assert sorted(x.cls for x in d.symbols).count("ankle") == 2  # A's and V's left port
    assert len(d.edges) == 3


def test_confidence_fit_recovers_signal() -> None:
    from pid.graph.confidence import FEATURES, expected_calibration_error, fit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, len(FEATURES)))
    true_p = 1 / (1 + np.exp(-(2.0 * X[:, 2] - 1.0 * X[:, 5])))  # fill up, slack down
    y = (rng.random(2000) < true_p).astype(float)
    m = fit(X, y, l2=0.1)
    assert m.coef[2] > 1.0 and m.coef[5] < -0.5
    ece, _ = expected_calibration_error(m.predict(X), y)
    assert ece < 0.05


@pytest.mark.xfail(reason="diagonal lines are not extracted (orthogonal-only line stage, "
                          "DECISIONS D-021); 6% of OPEN100 structural edges", strict=True)
def test_diagonal_line(pcfg: dict[str, Any]) -> None:
    img = canvas()
    cv2.line(img, (100, 100), (500, 400), 0, 5)
    s = [sym("A", 60, 60, 100, 100), sym("B", 500, 400, 540, 440)]
    d = run(img, s, pcfg)
    assert len(d.edges) >= 1


def test_text_next_to_line_adds_no_nodes(pcfg: dict[str, Any]) -> None:
    img = canvas()
    hline(img, 300, 100, 700)
    cv2.putText(img, "SIZE-RCS-100078-SPEC", (180, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    s = [sym("A", 60, 280, 100, 320), sym("B", 700, 280, 740, 320)]
    d = run(img, s, pcfg)
    assert len(d.edges) == 3 and not any(x.cls == "crossing" for x in d.symbols)


def test_ink_threshold_caps_below_gray_paper() -> None:
    from pid.lines.extract import ink_threshold

    cfg = {"threshold": 180, "paper_margin": 25}
    assert ink_threshold(np.full((64, 64), 255, np.uint8), cfg) == 180
    assert ink_threshold(np.full((64, 64), 186, np.uint8), cfg) == 161  # degraded gray paper
    assert ink_threshold(np.full((64, 64), 186, np.uint8), {"threshold": 180}) == 180
