"""Break down edge errors of a prediction run by cause.

  python scripts/analyze_errors.py --pred-dir outputs/preds/<run> --split dataset_pid_val

Uses the same node matching as the primary metric (greedy IoU >= 0.5). Reports:
- recall per GT line type;
- FN edges: endpoint node missing (by node kind) vs both endpoints found but unconnected;
- FP edges: spurious endpoint (by node kind) vs both matched but no GT edge;
- unmatched GT nodes by kind, and the distance from each to its nearest predicted node
  of any kind (localization misses vs. truly missing nodes).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from pid.config import load_config, resolve
from pid.eval.match import greedy_match
from pid.export.graphml import read_graphml
from pid.io.pid2graph import load_drawing, ref_from_uid

STRUCT = ("crossing", "ankle")


def kind(cls: str) -> str:
    return cls if cls in STRUCT else "symbol"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", action="append", required=True)
    args = ap.parse_args()
    cfg = load_config("data")
    for split in args.split:
        c: Counter[str] = Counter()
        miss_dist: list[float] = []
        for uid in (resolve(cfg["splits"]["dir"]) / f"{split}.txt").read_text().split():
            gt = load_drawing(ref_from_uid(uid, cfg), cfg, with_size=False)
            pred = read_graphml(Path(args.pred_dir) / f"{uid}.graphml", cfg["label_map"])
            m = greedy_match(gt.symbols, pred.symbols)  # pred -> gt
            inv = {g: p for p, g in m.items()}
            gs, ps = gt.symbol_by_id(), pred.symbol_by_id()
            pred_keys = {e.key() for e in pred.edges}
            gt_keys = {e.key() for e in gt.edges}
            for e in gt.edges:
                c[f"gt_{e.line_type.value}"] += 1
                pu, pv = inv.get(e.source), inv.get(e.target)
                if pu and pv and frozenset((pu, pv)) in pred_keys:
                    c[f"tp_{e.line_type.value}"] += 1
                elif pu is None or pv is None:
                    missing = [n for n, p in ((e.source, pu), (e.target, pv)) if p is None]
                    c["fn_node_missing:" + "+".join(sorted(kind(gs[n].cls) for n in missing))] += 1
                else:
                    c[f"fn_unconnected_{e.line_type.value}"] += 1
            for e in pred.edges:
                gu, gv = m.get(e.source), m.get(e.target)
                if gu and gv and frozenset((gu, gv)) in gt_keys:
                    continue
                if gu is None or gv is None:
                    spur = [n for n, g in ((e.source, gu), (e.target, gv)) if g is None]
                    c["fp_spurious_node:" + "+".join(sorted(kind(ps[n].cls) for n in spur))] += 1
                else:
                    c["fp_no_gt_edge"] += 1
            centers = np.array([p.bbox.center for p in pred.symbols]) if pred.symbols else None
            for g in gt.symbols:
                if g.id not in inv:
                    c[f"gt_node_unmatched:{kind(g.cls)}"] += 1
                    if centers is not None and kind(g.cls) != "symbol":
                        miss_dist.append(float(np.min(np.abs(centers - g.bbox.center).max(1))))
        print(f"== {split}")
        for lt in ("solid", "non-solid"):
            n = c[f"gt_{lt}"]
            if n:
                print(f"  recall {lt:9s} {c[f'tp_{lt}'] / n:.3f}  ({c[f'tp_{lt}']}/{n})")
        for k in sorted(k for k in c if k.startswith(("fn_", "fp_", "gt_node"))):
            print(f"  {k:45s} {c[k]}")
        if miss_dist:
            d = np.array(miss_dist)
            print("  unmatched structural GT node -> nearest pred node (L-inf px): "
                  f"<=4: {(d <= 4).mean():.2f}  4-12: {((d > 4) & (d <= 12)).mean():.2f}  "
                  f">12: {(d > 12).mean():.2f}  (n={len(d)})")


if __name__ == "__main__":
    main()
