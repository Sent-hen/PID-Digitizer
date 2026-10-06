"""Categorized error analysis of the final test predictions (Milestone 5).

  python scripts/error_analysis.py [--run final_test] [--examples 4]

Writes docs/error_analysis_counts.json and example crops to docs/error_examples/,
which docs/ERROR_ANALYSIS.md references. Categories (node matching = greedy IoU >= 0.5,
as in the primary metric):

* missed_symbol     GT symbol with no prediction overlapping it (any class)
* class_confusion   GT symbol matched class-agnostically but with the wrong class
* false_symbol      predicted symbol matching no GT symbol
* broken_pipe       GT edge missed: both endpoints found but not connected, or a
                    structural endpoint missing (line not extracted / joined)
* spurious_link     predicted edge with no GT edge (between matched nodes, or through
                    a spurious structural node)
* dashed_error      GT non-solid edge missed, or a found edge with the wrong line type
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pid.config import load_config, resolve
from pid.eval.match import greedy_match
from pid.export.graphml import read_graphml
from pid.io.drawing import Drawing
from pid.io.pid2graph import load_drawing, ref_from_uid
from pid.pipeline import load_gray

STRUCT = ("crossing", "ankle")


def analyze(gt: Drawing, pred: Drawing) -> dict[str, list[tuple[float, float]]]:
    """Return category -> list of (x, y) locations (original pixels)."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    gsym = [s for s in gt.symbols if s.cls not in STRUCT]
    psym = [s for s in pred.symbols if s.cls not in STRUCT]
    ms = greedy_match(gsym, psym)  # pred -> gt (symbols only, class-agnostic)
    gs = {s.id: s for s in gsym}
    for pid_, gid in ms.items():
        p = next(s for s in psym if s.id == pid_)
        if p.cls != gs[gid].cls:
            out["class_confusion"].append(gs[gid].bbox.center)
    matched_g = set(ms.values())
    out["missed_symbol"] += [s.bbox.center for s in gsym if s.id not in matched_g]
    out["false_symbol"] += [s.bbox.center for s in psym if s.id not in ms]

    m = greedy_match(gt.symbols, pred.symbols)  # all nodes, for edges
    inv = {g: p for p, g in m.items()}
    G, P = gt.symbol_by_id(), pred.symbol_by_id()
    pk = {e.key(): e for e in pred.edges}
    gk = {e.key(): e for e in gt.edges}
    for e in gt.edges:
        pu, pv = inv.get(e.source), inv.get(e.target)
        mid = tuple(np.mean([G[e.source].bbox.center, G[e.target].bbox.center], axis=0))
        hit = pu and pv and frozenset((pu, pv)) in pk
        if not hit:
            out["broken_pipe"].append(mid)  # type: ignore[arg-type]
            if e.line_type.value == "non-solid":
                out["dashed_error"].append(mid)  # type: ignore[arg-type]
        elif pk[frozenset((pu, pv))].line_type != e.line_type:  # type: ignore[arg-type]
            out["dashed_error"].append(mid)  # type: ignore[arg-type]
    for e in pred.edges:
        gu, gv = m.get(e.source), m.get(e.target)
        if not (gu and gv and frozenset((gu, gv)) in gk):
            mid = tuple(np.mean([P[e.source].bbox.center, P[e.target].bbox.center], axis=0))
            out["spurious_link"].append(mid)  # type: ignore[arg-type]
    return out


def crop(gt: Drawing, pred: Drawing, xy: tuple[float, float], path: Path, half: int) -> None:
    img = cv2.cvtColor(load_gray(gt.image_path or ""), cv2.COLOR_GRAY2BGR)
    for d, col, th in ((gt, (0, 0, 230), 3), (pred, (230, 120, 0), 1)):
        S = d.symbol_by_id()
        for e in d.edges:
            a, b = S[e.source].bbox.center, S[e.target].bbox.center
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, th)
        for s in d.symbols:
            b = s.bbox
            if s.cls in STRUCT:
                cv2.circle(img, (int(b.center[0]), int(b.center[1])), 5, col, th)
            else:
                cv2.rectangle(img, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), col, th)
    x, y = int(xy[0]), int(xy[1])
    h, w = img.shape[:2]
    x0, y0 = max(0, min(w - 2 * half, x - half)), max(0, min(h - 2 * half, y - half))
    c = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    cv2.circle(c, (x - x0, y - y0), 22, (0, 200, 255), 2)  # the error location
    cv2.imwrite(str(path), c, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final_test")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg, fe = load_config("data"), load_config("final_eval")
    counts: dict[str, dict[str, int]] = {}
    pool: dict[str, list[tuple[str, tuple[float, float]]]] = defaultdict(list)
    totals: dict[str, dict[str, int]] = {}
    for split in fe["test_splits"]:
        c: Counter[str] = Counter()
        tot: Counter[str] = Counter()
        for uid in (resolve(cfg["splits"]["dir"]) / f"{split}.txt").read_text().split():
            gt = load_drawing(ref_from_uid(uid, cfg), cfg)
            pred = read_graphml(resolve("outputs/preds") / args.run / f"{uid}.graphml",
                                cfg["label_map"])
            res = analyze(gt, pred)
            for k, v in res.items():
                c[k] += len(v)
                pool[k] += [(uid, xy) for xy in v]
            tot["gt_symbols"] += sum(s.cls not in STRUCT for s in gt.symbols)
            tot["gt_edges"] += len(gt.edges)
            tot["gt_nonsolid_edges"] += sum(e.line_type.value == "non-solid" for e in gt.edges)
            tot["pred_symbols"] += sum(s.cls not in STRUCT for s in pred.symbols)
            tot["pred_edges"] += len(pred.edges)
        counts[split], totals[split] = dict(c), dict(tot)
        print(split, dict(c), dict(tot))
    ex_dir = resolve("docs/error_examples")
    ex_dir.mkdir(parents=True, exist_ok=True)
    for f in ex_dir.glob("*.jpg"):
        f.unlink()
    rng = np.random.default_rng(args.seed)
    examples: dict[str, list[dict[str, Any]]] = {}
    for cat, items in sorted(pool.items()):
        picks = rng.choice(len(items), size=min(args.examples, len(items)), replace=False)
        examples[cat] = []
        for j, k in enumerate(sorted(picks)):
            uid, xy = items[int(k)]
            gt = load_drawing(ref_from_uid(uid, cfg), cfg)
            pred = read_graphml(resolve("outputs/preds") / args.run / f"{uid}.graphml",
                                cfg["label_map"])
            half = 160 if uid.startswith("open100") else 260
            name = f"{cat}_{j + 1}.jpg"
            crop(gt, pred, xy, ex_dir / name, half)
            examples[cat].append({"file": name, "uid": uid, "x": round(xy[0]), "y": round(xy[1])})
    resolve("docs/error_analysis_counts.json").write_text(json.dumps(
        {"run": args.run, "counts": counts, "totals": totals, "examples": examples}, indent=1))
    print("wrote docs/error_analysis_counts.json and", sum(map(len, examples.values())),
          "example crops")


if __name__ == "__main__":
    main()
