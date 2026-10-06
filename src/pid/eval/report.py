"""One entry point that computes every metric for a set of (gt, pred) drawings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pid.eval.connectivity import edge_map_pid2graph, edge_prf_dataset
from pid.eval.detection import node_ap, symbol_map
from pid.io.drawing import Drawing


def evaluate(pairs: Sequence[tuple[Drawing, Drawing]], cfg: Mapping[str, Any],
             iou_thr: float = 0.5) -> dict[str, Any]:
    """`cfg` is the data config (needs `symbol_classes` and `structural_classes`)."""
    for gt, pred in pairs:
        if gt.id != pred.id or gt.subset != pred.subset:
            raise ValueError(f"misaligned pair: {gt.subset}/{gt.id} vs {pred.subset}/{pred.id}")
    sym_classes = list(cfg["symbol_classes"])
    all_classes = sym_classes + list(cfg["structural_classes"])
    total, per = edge_prf_dataset(pairs, iou_thr)
    f1s = np.array([c.f1 for c in per]) if per else np.array([np.nan])
    return {
        "n_drawings": len(pairs),
        "symbols": symbol_map(pairs, sym_classes),
        "node_AP50": node_ap(pairs, all_classes, iou_thr),
        "edges": total.summary(),
        "edges_per_drawing_f1": {
            "mean": float(np.mean(f1s)),
            "median": float(np.median(f1s)),
            "p10": float(np.percentile(f1s, 10)),
        },
        "pid2graph_edge_map": edge_map_pid2graph(pairs),
    }
