"""Run the Milestone 5 evaluation matrix (configs/final_eval.yaml), sequentially.

  python scripts/final_eval.py [--only RUN ...] [--force]

Each run calls scripts/run_baseline.py once on the test splits. A run is skipped if all
its metrics files exist and were produced by the expected weights (sha1), so the
script can be resumed after an interruption without re-running test evaluations.
Runs execute one at a time: concurrent GPU/CPU jobs slowed each other ~20x (D-019).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from pid.config import load_config, resolve


def done(run: str, splits: list[str], sha: str, oracle: bool) -> bool:
    for s in splits:
        f = resolve("outputs/preds") / run / f"metrics_{s}.json"
        if not f.exists():
            return False
        m = json.loads(f.read_text())
        if not oracle and m.get("weights_sha1") != sha:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config("final_eval")
    splits = list(cfg["test_splits"])
    for run, spec in cfg["runs"].items():
        spec = spec or {}
        if args.only and run not in args.only:
            continue
        oracle = bool(spec.get("oracle_symbols"))
        if not args.force and done(run, splits, cfg["weights_sha1"], oracle):
            print(f"skip {run} (metrics exist)")
            continue
        cmd = [sys.executable, str(resolve("scripts/run_baseline.py")), "--run", run]
        for s in splits:
            cmd += ["--split", s]
        if oracle:
            cmd.append("--oracle-symbols")
        else:
            cmd += ["--weights", str(cfg["weights"])]
        if spec.get("degrade"):
            cmd += ["--degrade", str(spec["degrade"])]
        if spec.get("dexpi"):
            cmd.append("--dexpi")
        for kv in spec.get("set", []):
            cmd += ["--set", kv]
        print("run", run, " ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, check=True, cwd=resolve("."))


if __name__ == "__main__":
    main()
