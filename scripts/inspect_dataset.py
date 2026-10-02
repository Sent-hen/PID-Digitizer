"""Dataset audit: per-subset statistics, annotation anomalies, and sample overlays.

Usage: python scripts/inspect_dataset.py [--config configs/data.yaml] [--samples 5]
Writes docs/dataset_stats.json, docs/dataset_stats.md and docs/dataset_samples/*.jpg.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image

from pid.config import load_config, resolve
from pid.io.drawing import Drawing
from pid.io.overlay import render_overlay
from pid.io.pid2graph import DrawingRef, graph_to_drawing, iter_refs

Image.MAX_IMAGE_PIXELS = None


def describe(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "median": float(np.median(a)), "max": float(a.max())}


def audit_one(ref: DrawingRef, cfg: dict[str, Any]) -> tuple[Drawing, dict[str, Any]]:
    G = nx.read_graphml(ref.graphml)
    swapped = sum(1 for _, d in G.nodes(data=True)
                  if d["xmin"] > d["xmax"] or d["ymin"] > d["ymax"])
    drawing = graph_to_drawing(G, ref.drawing_id, cfg["label_map"], ref.subset)
    with Image.open(ref.image) as im:
        drawing.width, drawing.height = im.size
        mode = im.mode
    drawing.image_path = str(ref.image)
    W, H = drawing.width, drawing.height
    deg: Counter[str] = Counter()
    for s in drawing.symbols:
        deg[s.id] = 0
    for e in drawing.edges:
        deg[e.source] += 1
        deg[e.target] += 1
    sg = nx.Graph()
    sg.add_nodes_from(s.id for s in drawing.symbols)
    sg.add_edges_from((e.source, e.target) for e in drawing.edges)
    info = {
        "uid": ref.uid,
        "size": (W, H),
        "mode": mode,
        "ext": ref.image.suffix,
        "n_nodes": len(drawing.symbols),
        "n_edges": len(drawing.edges),
        "n_ignore": len(drawing.ignore_regions),
        "swapped_bbox": swapped,
        "out_of_image": sum(1 for s in drawing.symbols
                            if s.bbox.xmin < -1 or s.bbox.ymin < -1
                            or s.bbox.xmax > W + 1 or s.bbox.ymax > H + 1),
        "isolated_nodes": sum(1 for v in deg.values() if v == 0),
        "components": nx.number_connected_components(sg),
        "self_loops": nx.number_of_selfloops(sg),
    }
    return drawing, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(args.out)
    (out / "dataset_samples").mkdir(parents=True, exist_ok=True)

    per_subset: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "infos": [], "cls": Counter(), "lt": Counter(), "deg": defaultdict(Counter),
        "wh": defaultdict(list)})
    drawings: dict[str, Drawing] = {}
    for ref in iter_refs(cfg):
        d, info = audit_one(ref, cfg)
        drawings[ref.uid] = d
        S = per_subset[ref.subset]
        S["infos"].append(info)
        S["cls"].update(s.cls for s in d.symbols)
        S["lt"].update(e.line_type.value for e in d.edges)
        degree: Counter[str] = Counter()
        for e in d.edges:
            degree[e.source] += 1
            degree[e.target] += 1
        for s in d.symbols:
            S["deg"][s.cls][degree[s.id]] += 1
            S["wh"][s.cls].append((s.bbox.width, s.bbox.height))
        print(f"\r{ref.uid:30s}", end="", flush=True)
    print()

    stats: dict[str, Any] = {}
    for sub, S in per_subset.items():
        infos = S["infos"]
        stats[sub] = {
            "n_drawings": len(infos),
            "image_ext": dict(Counter(i["ext"] for i in infos)),
            "image_mode": dict(Counter(i["mode"] for i in infos)),
            "image_sizes": {f"{w}x{h}": c for (w, h), c in
                            Counter(tuple(i["size"]) for i in infos).most_common()},
            "nodes_per_drawing": describe([i["n_nodes"] for i in infos]),
            "edges_per_drawing": describe([i["n_edges"] for i in infos]),
            "class_counts": dict(S["cls"].most_common()),
            "line_types": dict(S["lt"].most_common()),
            "degree_by_class": {c: dict(sorted(v.items())) for c, v in S["deg"].items()},
            "median_wh_by_class": {c: [float(x) for x in np.median(np.array(v), axis=0)]
                                   for c, v in S["wh"].items()},
            "anomalies": {k: int(sum(i[k] for i in infos)) for k in
                          ("swapped_bbox", "out_of_image", "isolated_nodes", "self_loops")},
            "drawings_with_multiple_components": sum(1 for i in infos if i["components"] > 1),
            "ignore_regions_per_drawing": dict(Counter(i["n_ignore"] for i in infos)),
        }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "dataset_stats.md").write_text(to_markdown(stats, cfg), encoding="utf-8")

    rng = random.Random(args.seed)
    uids = sorted(drawings)
    picks = rng.sample(uids, args.samples)
    for sub in per_subset:  # guarantee at least one per subset
        if not any(p.startswith(sub + "/") for p in picks):
            picks.append(rng.choice([u for u in uids if u.startswith(sub + "/")]))
    for uid in picks:
        d = drawings[uid]
        name = uid.replace("/", "_")
        render_overlay(d, max_side=2000).save(out / "dataset_samples" / f"{name}.jpg",
                                              quality=85)
        save_crop(d, out / "dataset_samples" / f"{name}_crop.jpg", rng)
        print("overlay", uid)


def save_crop(d: Drawing, path: Path, rng: random.Random, size: int = 1200) -> None:
    """Native-resolution crop centered on a random non-structural symbol."""
    assert d.image_path is not None and d.width and d.height
    syms = [s for s in d.symbols if s.cls not in ("crossing", "ankle")] or d.symbols
    cx, cy = rng.choice(syms).bbox.center
    x0 = int(min(max(0, cx - size / 2), max(0, d.width - size)))
    y0 = int(min(max(0, cy - size / 2), max(0, d.height - size)))
    img = Image.open(d.image_path).crop((x0, y0, x0 + size, y0 + size))
    shifted = d.model_copy(deep=True)
    for s in shifted.symbols:
        b = s.bbox
        b.xmin, b.xmax, b.ymin, b.ymax = b.xmin - x0, b.xmax - x0, b.ymin - y0, b.ymax - y0
    shifted.ignore_regions = []
    render_overlay(shifted, image=img, max_side=size).save(path, quality=90)


def to_markdown(stats: dict[str, Any], cfg: dict[str, Any]) -> str:
    subs = list(stats)
    lines = ["# PID2Graph dataset statistics", "",
             "Generated by `scripts/inspect_dataset.py`; do not edit by hand.", "",
             "| | " + " | ".join(subs) + " |", "|---|" + "---|" * len(subs)]

    def row(name: str, f: Any) -> None:
        lines.append(f"| {name} | " + " | ".join(str(f(stats[s])) for s in subs) + " |")

    row("drawings", lambda s: s["n_drawings"])
    row("image sizes (distinct)", lambda s: len(s["image_sizes"]))
    row("most common size", lambda s: next(iter(s["image_sizes"])))
    row("nodes / drawing (mean ± std)",
        lambda s: f"{s['nodes_per_drawing']['mean']:.0f} ± {s['nodes_per_drawing']['std']:.0f}")
    row("edges / drawing (mean ± std)",
        lambda s: f"{s['edges_per_drawing']['mean']:.0f} ± {s['edges_per_drawing']['std']:.0f}")
    for k in ("swapped_bbox", "out_of_image", "isolated_nodes", "self_loops"):
        row(f"anomaly: {k}", lambda s, k=k: s["anomalies"][k])
    row("drawings with >1 component", lambda s: s["drawings_with_multiple_components"])
    lines += ["", "## Node class counts", "", "| class | " + " | ".join(subs) + " |",
              "|---|" + "---|" * len(subs)]
    for c in cfg["symbol_classes"] + cfg["structural_classes"]:
        lines.append(f"| {c} | " + " | ".join(str(stats[s]["class_counts"].get(c, 0))
                                               for s in subs) + " |")
    lines += ["", "## Median symbol size (w × h px)", "", "| class | " + " | ".join(subs) + " |",
              "|---|" + "---|" * len(subs)]
    for c in cfg["symbol_classes"] + cfg["structural_classes"]:
        cells = []
        for s in subs:
            wh = stats[s]["median_wh_by_class"].get(c)
            cells.append(f"{wh[0]:.0f} × {wh[1]:.0f}" if wh else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Edge line types", "", "| type | " + " | ".join(subs) + " |",
              "|---|" + "---|" * len(subs)]
    for t in ("solid", "non-solid"):
        lines.append(f"| {t} | " + " | ".join(str(stats[s]["line_types"].get(t, 0))
                                               for s in subs) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
