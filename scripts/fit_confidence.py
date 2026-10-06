"""Fit and evaluate the edge-confidence model on validation drawings.

  python scripts/fit_confidence.py --split dataset_pid_val --split synthetic_val

1. Run the pipeline (detector from detect.yaml, confidence disabled) on every drawing,
   keeping per-edge features; label each predicted edge 1 if it matches a GT edge.
2. 2-fold cross-validation by drawing: fit on one half, score the other, so the
   reported calibration (ECE) and edge mAP are out-of-sample.
3. Fit on all drawings, save configs/confidence_model.json, and write the reliability
   diagram to docs/reliability_diagram.png and a JSON summary to outputs/.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from pid.config import apply_overrides, load_config, resolve
from pid.eval.connectivity import edge_map_pid2graph
from pid.eval.match import greedy_match
from pid.graph.confidence import (
    apply_confidence,
    expected_calibration_error,
    fit,
    vectorize,
)
from pid.io.drawing import Drawing
from pid.io.pid2graph import load_drawing, ref_from_uid
from pid.pipeline import digitize


def labels_for(gt: Drawing, pred: Drawing) -> list[float]:
    m = greedy_match(gt.symbols, pred.symbols)
    gt_keys = {e.key() for e in gt.edges}
    out = []
    for e in pred.edges:
        gu, gv = m.get(e.source), m.get(e.target)
        out.append(float(bool(gu and gv and frozenset((gu, gv)) in gt_keys)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", default="configs/confidence_model.json")
    args = ap.parse_args()
    from pid.detect.model import TiledDetector

    pcfg = apply_overrides(load_config("pipeline"), ["confidence.enabled=false"])
    dcfg, det_cfg = load_config("data"), load_config("detect")
    weights = resolve(args.weights or det_cfg["train"]["out"])
    detector = TiledDetector(det_cfg, weights)

    items: list[dict[str, Any]] = []  # per drawing: gt, pred, X, y
    for split in args.split:
        for uid in (resolve(dcfg["splits"]["dir"]) / f"{split}.txt").read_text().split():
            gt = load_drawing(ref_from_uid(uid, dcfg), dcfg)
            feats: dict[str, dict[str, float]] = {}
            pred = digitize(gt.image_path or "", pcfg, gt.id, gt.subset, detector=detector,
                            features_out=feats)
            X = vectorize([feats[e.id] for e in pred.edges])
            items.append({"uid": uid, "gt": gt, "pred": pred, "feats": feats, "X": X,
                          "y": np.array(labels_for(gt, pred))})
        print(f"collected {split}")

    folds = [items[0::2], items[1::2]]
    p_all, y_all, pairs_oos = [], [], []
    for k in (0, 1):
        train, test = folds[1 - k], folds[k]
        model = fit(np.vstack([it["X"] for it in train]),
                    np.concatenate([it["y"] for it in train]), l2=args.l2)
        for it in test:
            d = apply_confidence(it["pred"].model_copy(deep=True), it["feats"], model)
            p_all.append(np.array([e.score for e in d.edges]))
            y_all.append(it["y"])
            pairs_oos.append((it["gt"], d))
    p, y = np.concatenate(p_all), np.concatenate(y_all)
    ece, table = expected_calibration_error(p, y)
    base = edge_map_pid2graph([(it["gt"], it["pred"]) for it in items])  # all scores 1.0
    oos = edge_map_pid2graph(pairs_oos)

    model = fit(np.vstack([it["X"] for it in items]),
                np.concatenate([it["y"] for it in items]), l2=args.l2)
    out = resolve(args.out)
    model.save(out)
    summary = {"splits": args.split, "weights": str(weights.name), "n_edges": int(len(y)),
               "positive_rate": float(y.mean()), "cv_ece": ece, "reliability": table,
               "edge_map_constant_scores": base, "edge_map_cv_confidence": oos,
               "coef": dict(zip(model.features, model.coef, strict=True)),
               "intercept": model.intercept}
    resolve("outputs").mkdir(exist_ok=True)
    (resolve("outputs") / "confidence_fit.json").write_text(json.dumps(summary, indent=1))
    plot(table, ece, resolve("docs/reliability_diagram.png"))
    print(json.dumps({k: v for k, v in summary.items() if k != "reliability"}, indent=1))


def plot(table: list[dict[str, float]], ece: float, path: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    conf = [t["confidence"] for t in table]
    acc = [t["accuracy"] for t in table]
    ax.plot(conf, acc, marker="o", color="#2a6fdb", lw=1.5, label="edge confidence (2-fold CV)")
    for c, a, t in zip(conf, acc, table, strict=True):
        ax.annotate(str(t["n"]), (c, a), textcoords="offset points", xytext=(4, -10),
                    fontsize=6, color="#555555")
    ax.set_xlabel("predicted probability edge is correct")
    ax.set_ylabel("observed fraction correct")
    ax.set_title(f"Reliability (val), ECE = {ece:.3f}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
