"""Download the detector weights from the GitHub release and verify their SHA-1.

  python scripts/fetch_weights.py [--dest weights/detector_m2c.pt]

Standard library only, so it runs before the project is installed (e.g. before
`docker compose up`). Sources, in order:
  1. the public release URL (works if the repository is public);
  2. the GitHub API with a token from $GITHUB_TOKEN / $GH_TOKEN (private repository);
  3. `gh release download` (GitHub CLI, already authenticated).
The file is only moved into place after its SHA-1 matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "Sent-hen/PID-Digitizer"
TAG = "weights-detector-m2c-v1"
ASSET = "detector_m2c.pt"
SHA1 = "531ab26ce150178a13c50be344109ec9db210605"
ROOT = Path(__file__).resolve().parents[1]


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out: Path, headers: dict[str, str]) -> None:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r, out.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def via_public(out: Path) -> bool:
    url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
    try:
        download(url, out, {})
        return True
    except urllib.error.HTTPError as exc:
        print(f"public download failed ({exc.code}); the repository may be private")
        return False


def via_api(out: Path) -> bool:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return False
    auth = {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}",
        headers={**auth, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            release = json.load(r)
        asset = next(a for a in release["assets"] if a["name"] == ASSET)
        download(asset["url"], out, {**auth, "Accept": "application/octet-stream"})
        return True
    except (urllib.error.HTTPError, StopIteration) as exc:
        print(f"API download failed: {exc}")
        return False


def via_gh(out: Path) -> bool:
    if shutil.which("gh") is None:
        return False
    r = subprocess.run(["gh", "release", "download", TAG, "--repo", REPO, "--pattern", ASSET,
                        "--output", str(out), "--clobber"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"gh download failed: {r.stderr.strip()}")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "weights" / ASSET))
    args = ap.parse_args()
    dest = Path(args.dest)
    if dest.exists() and sha1(dest) == SHA1:
        print(f"{dest} already present (sha1 ok)")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        part = Path(tmp) / ASSET
        for fetch in (via_public, via_api, via_gh):
            if fetch(part) and part.exists():
                got = sha1(part)
                if got == SHA1:
                    part.replace(dest)
                    print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB, sha1 ok)")
                    return 0
                print(f"sha1 mismatch ({got}); discarding")
    print("could not download the weights: make the repository public, set GITHUB_TOKEN, "
          f"or log in with `gh auth login` (release {TAG} of {REPO}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
