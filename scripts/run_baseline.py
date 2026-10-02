"""Run the walking-skeleton pipeline over split lists and evaluate.

  python scripts/run_baseline.py --run m1 --split dataset_pid_val ...
  python scripts/run_baseline.py --run m1_oracle --oracle-symbols --split ...   # GT symbols

Every drawing is processed; failures are caught, logged and counted (a failed drawing
is scored as an empty prediction). Predictions go to outputs/preds/<run>/<uid>.json,
minimal JSON exports to outputs/export/<run>/<uid>.json, metrics to
outputs/eval_log.jsonl and outputs/preds/<run>/metrics_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any

from pid.config import load_config, resolve
from pid.eval.report import evaluate
from pid.export.json_export import export_minimal
from pid.io.drawing import Drawing
from pid.io.pid2graph import load_drawing, ref_from_uid
from pid.pipeline import Detector, digitize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--oracle-symbols", action="store_true",
                    help="use GT symbols instead of the detector (error-propagation analysis)")
    ap.add_argument("--weights", help="detector weights (default: detect.yaml train.out)")
    args = ap.parse_args()
    pcfg, dcfg = load_config("pipeline"), load_config("data")

    detector: Detector | None = None
    if not args.oracle_symbols:
        from pid.detect.model import TiledDetector

        det_cfg = load_config("detect")
        detector = TiledDetector(det_cfg, resolve(args.weights or det_cfg["train"]["out"]))

    pred_root = resolve("outputs/preds") / args.run
    export_root = resolve("outputs/export") / args.run
    for split in args.split:
        uids = (resolve(dcfg["splits"]["dir"]) / f"{split}.txt").read_text().split()
        pairs: list[tuple[Drawing, Drawing]] = []
        failures: list[dict[str, str]] = []
        times: list[float] = []
        for uid in uids:
            gt = load_drawing(ref_from_uid(uid, dcfg), dcfg)
            t0 = time.perf_counter()
            try:
                pred = digitize(gt.image_path or "", pcfg, gt.id, gt.subset, detector=detector,
                                oracle_symbols=gt.symbols if args.oracle_symbols else None)
            except Exception as exc:  # noqa: BLE001 - every failure is counted, never fatal
                failures.append({"uid": uid, "error": repr(exc),
                                 "trace": traceback.format_exc(limit=3)})
                print(f"FAILED {uid}: {exc!r}")
                pred = Drawing(id=gt.id, subset=gt.subset)
            times.append(time.perf_counter() - t0)
            out = pred_root / f"{uid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(pred.model_dump_json(), encoding="utf-8")
            exp = export_root / f"{uid}.json"
            exp.parent.mkdir(parents=True, exist_ok=True)
            exp.write_text(json.dumps(export_minimal(pred)), encoding="utf-8")
            pairs.append((gt, pred))
        res: dict[str, Any] = evaluate(pairs, dcfg)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": args.run, "split": split,
                  "oracle_symbols": args.oracle_symbols, "n_failures": len(failures),
                  "failures": failures, "sec_per_drawing": sum(times) / max(1, len(times)),
                  **res}
        (pred_root / f"metrics_{split}.json").write_text(json.dumps(record, indent=1))
        with resolve("outputs/eval_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        e, s = res["edges"], res["symbols"]
        print(f"{args.run} {split}: n={len(uids)} failures={len(failures)} "
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
