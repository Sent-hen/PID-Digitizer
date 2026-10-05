"""Run the walking-skeleton pipeline over split lists and evaluate.

  python scripts/run_baseline.py --run m1 --split dataset_pid_val ...
  python scripts/run_baseline.py --run m1_oracle --oracle-symbols --split ...   # GT symbols

Every drawing is processed; failures are caught, logged and counted (a failed drawing
is scored as an empty prediction). Predictions are GraphML (the project output
format) at outputs/preds/<run>/<uid>.graphml; metrics go to outputs/eval_log.jsonl and
outputs/preds/<run>/metrics_<split>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from typing import Any

from pid.config import apply_overrides, load_config, resolve
from pid.eval.report import evaluate
from pid.export.graphml import read_graphml, validate_graphml, write_graphml
from pid.io.drawing import Drawing
from pid.io.pid2graph import load_drawing, ref_from_uid
from pid.pipeline import Detector, digitize, load_gray
from pid.preprocess.degrade import degrade, transform_drawing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--oracle-symbols", action="store_true",
                    help="use GT symbols instead of the detector (error-propagation analysis)")
    ap.add_argument("--weights", help="detector weights (default: detect.yaml train.out)")
    ap.add_argument("--degrade", type=int, default=0, choices=[0, 1, 2, 3],
                    help="degradation severity (configs/degrade.yaml); GT is mapped to match")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override configs/pipeline.yaml, e.g. --set crop.enabled=true")
    args = ap.parse_args()
    pcfg = apply_overrides(load_config("pipeline"), args.set)
    dcfg = load_config("data")
    degcfg = load_config("degrade")

    detector: Detector | None = None
    if not args.oracle_symbols:
        from pid.detect.model import TiledDetector

        det_cfg = load_config("detect")
        detector = TiledDetector(det_cfg, resolve(args.weights or det_cfg["train"]["out"]))

    pred_root = resolve("outputs/preds") / args.run
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=resolve(".")).stdout.strip()
    cfg_hash = hashlib.sha1(json.dumps([pcfg, load_config("detect")], sort_keys=True,
                                       default=str).encode()).hexdigest()[:12]
    for split in args.split:
        uids = (resolve(dcfg["splits"]["dir"]) / f"{split}.txt").read_text().split()
        pairs: list[tuple[Drawing, Drawing]] = []
        failures: list[dict[str, str]] = []
        invalid: list[dict[str, Any]] = []
        times: list[float] = []
        for uid in uids:
            gt = load_drawing(ref_from_uid(uid, dcfg), dcfg)
            t0 = time.perf_counter()
            try:
                gray = None
                if args.degrade:
                    gray, M = degrade(load_gray(gt.image_path or ""), args.degrade, degcfg, uid)
                    gt = transform_drawing(gt, M, float(pcfg["graph"].get("struct_size", 8)))
                pred = digitize(gt.image_path or "", pcfg, gt.id, gt.subset, detector=detector,
                                oracle_symbols=gt.symbols if args.oracle_symbols else None,
                                gray=gray)
            except Exception as exc:  # noqa: BLE001 - every failure is counted, never fatal
                failures.append({"uid": uid, "error": repr(exc),
                                 "trace": traceback.format_exc(limit=3)})
                print(f"FAILED {uid}: {exc!r}")
                pred = Drawing(id=gt.id, subset=gt.subset)
            times.append(time.perf_counter() - t0)
            gpath = write_graphml(pred, pred_root / f"{uid}.graphml",
                                  pipeline_version=commit, config_hash=cfg_hash)
            # Acceptance: every output passes the GraphML XSD and reloads to the same graph.
            errs = validate_graphml(gpath.read_bytes())
            if errs or read_graphml(gpath, dcfg["label_map"]).edges != pred.edges:
                invalid.append({"uid": uid, "errors": errs[:3]})
            pairs.append((gt, pred))
        res: dict[str, Any] = evaluate(pairs, dcfg)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": args.run, "split": split,
                  "oracle_symbols": args.oracle_symbols, "overrides": args.set,
                  "degrade_level": args.degrade, "n_failures": len(failures),
                  "n_invalid_graphml": len(invalid), "invalid_graphml": invalid,
                  "failures": failures, "sec_per_drawing": sum(times) / max(1, len(times)),
                  **res}
        (pred_root / f"metrics_{split}.json").write_text(json.dumps(record, indent=1))
        with resolve("outputs/eval_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        e, s = res["edges"], res["symbols"]
        print(f"{args.run} {split}: n={len(uids)} failures={len(failures)} "
              f"invalid_graphml={len(invalid)} "
              f"symbol mAP50={s['mAP50']:.3f} mAP50:95={s['mAP50_95']:.3f} "
              f"nodeAP={res['node_AP50']:.3f} edge P/R/F1={e['precision']:.3f}/"
              f"{e['recall']:.3f}/{e['f1']:.3f} "
              f"(median {res['edges_per_drawing_f1']['median']:.3f}, "
              f"p10 {res['edges_per_drawing_f1']['p10']:.3f}) "
              f"lineAcc={e['line_type_accuracy']:.3f} "
              f"edge mAP={res['pid2graph_edge_map']['edge_mAP']:.3f} "
              f"{record['sec_per_drawing']:.1f}s/drawing")


if __name__ == "__main__":
    main()
