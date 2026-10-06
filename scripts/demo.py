"""Upload the bundled sample drawing to a running service and print where to review it.

  python scripts/demo.py [--url http://localhost:8000] [--image docker/sample/dataset_pid_10.png]

Standard library only. Waits for /health to report the model as loaded, uploads the
image, polls the job, and prints a summary plus the review-UI and download URLs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def upload(url: str, image: Path) -> Any:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{image.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{url}/jobs", data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--image", default=str(ROOT / "docker" / "sample" / "dataset_pid_10.png"))
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()
    url, t0 = args.url.rstrip("/"), time.time()

    print(f"waiting for {url}/health ...")
    while True:
        try:
            h = get(f"{url}/health")
            if h.get("model_loaded"):
                break
            if h.get("model_error"):
                print(f"model failed to load: {h['model_error']}", file=sys.stderr)
                return 1
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.time() - t0 > args.timeout:
            print("service did not become ready", file=sys.stderr)
            return 1
        time.sleep(3)

    job = upload(url, Path(args.image))
    print(f"uploaded {Path(args.image).name} -> job {job['id']}")
    while job["status"] not in ("done", "failed"):
        time.sleep(2)
        job = get(f"{url}/jobs/{job['id']}")
        print(f"  {job['status']} ({job['stage']})")
    if job["status"] == "failed":
        print(f"job failed: {job['error']}", file=sys.stderr)
        return 1
    print(f"done in {job['updated'] - job['created']:.1f} s: "
          f"{job['n_nodes']} nodes, {job['n_edges']} edges\n"
          f"  review UI : {url}/  (pick the job in 'Recent jobs')\n"
          f"  GraphML   : {url}/jobs/{job['id']}/result?format=graphml\n"
          f"  DEXPI XML : {url}/jobs/{job['id']}/result?format=dexpi\n"
          f"  API docs  : {url}/docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
