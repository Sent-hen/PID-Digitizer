"""Create leakage-controlled train/val/test splits.

Near-duplicate detection: each drawing is reduced to a coarse binary *ink map*
(cell = 1 if it contains any dark pixel), with the annotated ignore regions
(frame / title block, shared by every Dataset-PID sheet) masked out. Drawings in the
same subset whose ink maps have Jaccard similarity >= `ink_jaccard_min`, or whose
annotation graphs are identical, are unioned into one group. Groups (not drawings)
are split per subset into train/val/test with a fixed seed; subsets listed in
`held_out_subsets` go entirely to test.

(pHash was tried first and rejected: on sparse line drawings on white paper its
low-frequency DCT is near-constant, giving distance 0 for unrelated sheets. See
DECISIONS.md.)

Writes splits/<subset>_<split>.txt (one uid per line) and splits/groups.json.
Usage: python scripts/make_splits.py [--config data] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pid.config import load_config, resolve
from pid.io.pid2graph import DrawingRef, iter_refs, load_drawing

Image.MAX_IMAGE_PIXELS = None
GRID = (96, 64)  # ink-map cells (w, h); every subset is ~3:2


def fingerprint(ref: DrawingRef) -> tuple[str, NDArray[np.bool_], str]:
    """(uid, flattened ink map, annotation-graph hash)."""
    d = load_drawing(ref, with_size=False)
    with Image.open(ref.image) as im:
        a = np.asarray(im.convert("L")) < 128
    H, W = a.shape
    for r in d.ignore_regions:
        a[max(0, int(r.ymin)):int(r.ymax) + 1, max(0, int(r.xmin)):int(r.xmax) + 1] = False
    gw, gh = GRID
    ys = np.linspace(0, H, gh + 1).astype(int)
    xs = np.linspace(0, W, gw + 1).astype(int)
    rows = np.logical_or.reduceat(a, ys[:-1], axis=0)
    ink = np.logical_or.reduceat(rows, xs[:-1], axis=1)
    sig = sorted((s.cls, round(s.bbox.xmin), round(s.bbox.ymin)) for s in d.symbols)
    gh_ = hashlib.sha1(json.dumps(sig).encode()).hexdigest()
    return ref.uid, ink.ravel(), gh_


def jaccard_matrix(m: NDArray[np.bool_]) -> NDArray[np.float64]:
    f = m.astype(np.float64)
    inter = f @ f.T
    cnt = f.sum(1)
    union = cnt[:, None] + cnt[None, :] - inter
    out: NDArray[np.float64] = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return out


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.p = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg: dict[str, Any] = cfg["splits"]
    out = resolve(scfg["dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache = resolve(".cache/ink_fingerprints.npz")
    cache.parent.mkdir(parents=True, exist_ok=True)

    refs = list(iter_refs(cfg))
    inks: dict[str, NDArray[np.bool_]] = {}
    ghash: dict[str, str] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        for u, row, g in zip(z["uids"], z["inks"], z["ghash"], strict=True):
            inks[str(u)], ghash[str(u)] = row.astype(bool), str(g)
    todo = [r for r in refs if r.uid not in inks]
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            for n, (uid, ink, g) in enumerate(ex.map(fingerprint, todo, chunksize=4), 1):
                inks[uid], ghash[uid] = ink, g
                if n % 100 == 0:
                    print(f"fingerprinted {n}/{len(todo)}")
        uids_all = sorted(inks)
        np.savez_compressed(cache, uids=np.array(uids_all),
                            inks=np.stack([inks[u] for u in uids_all]),
                            ghash=np.array([ghash[u] for u in uids_all]))

    by_subset: dict[str, list[str]] = defaultdict(list)
    for r in refs:
        by_subset[r.subset].append(r.uid)

    jmin = float(scfg["ink_jaccard_min"])
    groups_out: dict[str, Any] = {"ink_jaccard_min": jmin, "grid": GRID, "subsets": {}}
    rng = random.Random(int(scfg["seed"]))
    for subset, uids in by_subset.items():
        J = jaccard_matrix(np.stack([inks[u] for u in uids]))
        np.fill_diagonal(J, 0.0)
        uf = UnionFind(uids)
        for i, j in zip(*np.nonzero(np.triu(J >= jmin, 1)), strict=True):
            uf.union(uids[i], uids[j])
        by_hash: dict[str, list[str]] = defaultdict(list)
        for u in uids:
            by_hash[ghash[u]].append(u)
        for members in by_hash.values():
            for u in members[1:]:
                uf.union(members[0], u)
        groups: dict[str, list[str]] = defaultdict(list)
        for u in uids:
            groups[uf.find(u)].append(u)
        glist = sorted(groups.values(), key=lambda g: int(g[0].split("/")[1]))
        sizes = Counter(len(g) for g in glist)
        nn = J.max(1)
        groups_out["subsets"][subset] = {
            "n_drawings": len(uids), "n_groups": len(glist),
            "group_size_hist": {str(k): v for k, v in sorted(sizes.items())},
            "nearest_neighbor_jaccard": {
                "min": float(nn.min()), "median": float(np.median(nn)),
                "p99": float(np.percentile(nn, 99)), "max": float(nn.max())},
            "duplicate_graphs": sum(len(m) - 1 for m in by_hash.values()),
            "multi_member_groups": [g for g in glist if len(g) > 1],
        }

        if subset in scfg["held_out_subsets"]:
            assign = {"test": [u for g in glist for u in g]}
        else:
            order = list(range(len(glist)))
            rng.shuffle(order)
            n = len(glist)
            n_test = round(n * scfg["test_fraction"])
            n_val = round(n * scfg["val_fraction"])
            parts = {"test": order[:n_test], "val": order[n_test:n_test + n_val],
                     "train": order[n_test + n_val:]}
            assign = {k: sorted((u for i in idx for u in glist[i]),
                                key=lambda u: int(u.split("/")[1]))
                      for k, idx in parts.items()}
        for split, members in assign.items():
            (out / f"{subset}_{split}.txt").write_text("\n".join(members) + "\n")
            print(f"{subset:12s} {split:5s} {len(members):4d}")
        s = groups_out["subsets"][subset]
        print(f"{subset}: {s['n_groups']} groups, sizes {s['group_size_hist']}, "
              f"NN Jaccard {s['nearest_neighbor_jaccard']}")
    (out / "groups.json").write_text(json.dumps(groups_out, indent=1))


if __name__ == "__main__":
    main()
