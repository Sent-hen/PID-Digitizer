"""Graph-level (edge) metrics.

Two metrics are provided:

* `edge_prf` - our primary connectivity metric. Nodes are matched by IoU (greedy,
  class-agnostic, IoU >= 0.5). A predicted edge is correct only if both endpoints are
  matched and the matched GT nodes are connected. Line type is ignored for P/R/F1 and
  reported separately as accuracy over correctly predicted edges.
* `edge_map_pid2graph` - the official PID2Graph edge mAP (Stürmer et al. 2025,
  Algorithm A1): Hungarian node matching on gIoU, per-edge-class AP, mean over classes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from pid.eval.ap import average_precision
from pid.eval.match import greedy_match, hungarian_match
from pid.io.drawing import Drawing, Edge, LineType


@dataclass
class EdgeCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    type_correct: int = 0  # among TP edges, line type also correct

    def __iadd__(self, other: EdgeCounts) -> EdgeCounts:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.type_correct += other.type_correct
        return self

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0 if self.fn == 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def line_type_accuracy(self) -> float:
        return self.type_correct / self.tp if self.tp else float("nan")

    def summary(self) -> dict[str, float]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "line_type_accuracy": self.line_type_accuracy}


def _gt_edge_index(gt: Drawing) -> dict[frozenset[str], Edge]:
    return {e.key(): e for e in gt.edges}


def edge_prf(gt: Drawing, pred: Drawing, iou_thr: float = 0.5,
             node_match: dict[str, str] | None = None) -> EdgeCounts:
    """Edge TP/FP/FN for one drawing. `node_match` ({pred_id: gt_id}) overrides matching."""
    m = node_match if node_match is not None else greedy_match(gt.symbols, pred.symbols, iou_thr)
    gt_edges = _gt_edge_index(gt)
    found: set[frozenset[str]] = set()
    c = EdgeCounts()
    for e in pred.edges:
        gu, gv = m.get(e.source), m.get(e.target)
        key = frozenset((gu, gv)) if gu is not None and gv is not None else None
        if key is not None and len(key) == 2 and key in gt_edges and key not in found:
            found.add(key)
            c.tp += 1
            if gt_edges[key].line_type == e.line_type:
                c.type_correct += 1
        else:
            c.fp += 1  # unmatched endpoint, non-edge, self-loop after matching, or duplicate
    c.fn = len(gt_edges) - len(found)
    return c


def edge_prf_dataset(pairs: Iterable[tuple[Drawing, Drawing]], iou_thr: float = 0.5,
                     ) -> tuple[EdgeCounts, list[EdgeCounts]]:
    """Micro-averaged totals plus per-drawing counts (for median/p10 reporting)."""
    total, per = EdgeCounts(), []
    for gt, pred in pairs:
        c = edge_prf(gt, pred, iou_thr)
        per.append(c)
        total += c
    return total, per


def edge_map_pid2graph(pairs: Iterable[tuple[Drawing, Drawing]],
                       classes: tuple[LineType, ...] = (LineType.SOLID, LineType.NON_SOLID),
                       min_iou: float | None = None) -> dict[str, float]:
    """Official PID2Graph edge mAP (Algorithm A1), accumulated over all drawings.

    Per class c: TPs are predicted edges whose Hungarian-mapped GT edge exists and has
    the same class; FPs are all other predicted edges; FNs are GT edges with no
    predicted edge between the mapped nodes. AP_c uses n_positives = TP_c + FN_c.
    Classes without positives are excluded from the mean.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    hits: dict[str, list[bool]] = defaultdict(list)
    npos: dict[str, int] = defaultdict(int)
    for gt, pred in pairs:
        m = hungarian_match(gt.symbols, pred.symbols, min_iou=min_iou)  # pred -> gt
        inv = {g: p for p, g in m.items()}
        gt_edges = _gt_edge_index(gt)
        pred_keys = {e.key() for e in pred.edges}
        for e in pred.edges:
            gu, gv = m.get(e.source), m.get(e.target)
            ge = gt_edges.get(frozenset((gu, gv))) if gu and gv and gu != gv else None
            ok = ge is not None and ge.line_type == e.line_type
            scores[e.line_type.value].append(e.score)
            hits[e.line_type.value].append(ok)
            if ok:
                npos[e.line_type.value] += 1
        for ge in gt.edges:
            pu, pv = inv.get(ge.source), inv.get(ge.target)
            if pu is None or pv is None or frozenset((pu, pv)) not in pred_keys:
                npos[ge.line_type.value] += 1  # FN
    out: dict[str, float] = {}
    aps = []
    for c in classes:
        ap = average_precision(scores[c.value], hits[c.value], npos[c.value])
        out[f"AP_{c.value}"] = ap
        if ap == ap:  # not NaN
            aps.append(ap)
    out["edge_mAP"] = sum(aps) / len(aps) if aps else float("nan")
    return out
