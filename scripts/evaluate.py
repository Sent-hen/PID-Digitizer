"""Evaluate predictions against PID2Graph ground truth.

Predictions are one GraphML file per drawing at <pred-dir>/<subset>/<id>.graphml.
`--oracle` evaluates ground truth against itself (harness sanity check; must be 1.0).

Usage:
  python scripts/evaluate.py --split dataset_pid_test --oracle
  python scripts/evaluate.py --split dataset_pid_test --pred-dir outputs/preds/run1
Results are printed and appended to outputs/eval_log.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from pid.config import load_config, resolve
from pid.eval.report import evaluate
from pid.export.graphml import read_graphml
from pid.io.drawing import Drawing
from pid.io.pid2graph import load_drawing, ref_from_uid


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=resolve(".")).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data")
    ap.add_argument("--split", action="append", required=True,
                    help="split list name under splits/, e.g. dataset_pid_test (repeatable)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred-dir")
    src.add_argument("--oracle", action="store_true")
    ap.add_argument("--log", default="outputs/eval_log.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for split in args.split:
        uids = (resolve(cfg["splits"]["dir"]) / f"{split}.txt").read_text().split()
        pairs: list[tuple[Drawing, Drawing]] = []
        missing = 0
        for uid in uids:
            gt = load_drawing(ref_from_uid(uid, cfg), cfg)
            if args.oracle:
                pred = gt.model_copy(deep=True)
            else:
                p = Path(args.pred_dir) / f"{uid}.graphml"
                if not p.exists():
                    missing += 1  # counts as an empty prediction, never silently skipped
                    pred = Drawing(id=gt.id, subset=gt.subset)
                else:
                    pred = read_graphml(p, cfg["label_map"])
            pairs.append((gt, pred))
        res = evaluate(pairs, cfg)
        record = {"time": dt.datetime.now().isoformat(timespec="seconds"), "commit": git_hash(),
                  "split": split, "source": "oracle" if args.oracle else args.pred_dir,
                  "missing_predictions": missing, **res}
        print(json.dumps(record, indent=1))
        log = resolve(args.log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
