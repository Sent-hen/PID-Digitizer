"""Loader for the PID2Graph dataset (Stürmer et al., 2025; Zenodo 14803338).

Each drawing is `<id>.png|jpg` + `<id>.graphml`. GraphML key ids differ between files
(e.g. `d1` is `xmin` in one file and `ymin` in another, and the same attribute may be
declared twice with long/double types), so attributes are resolved by name.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from pid.config import load_config, resolve
from pid.io.drawing import BBox, Drawing, Edge, LineType, Symbol

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class DrawingRef:
    subset: str  # canonical subset key, e.g. "dataset_pid"
    drawing_id: str  # file stem, e.g. "17"
    graphml: Path
    image: Path

    @property
    def uid(self) -> str:
        return f"{self.subset}/{self.drawing_id}"


def _image_for(graphml: Path) -> Path:
    for suf in IMAGE_SUFFIXES:
        p = graphml.with_suffix(suf)
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for {graphml}")


def _sort_key(p: Path) -> tuple[int, str]:
    return (int(p.stem), "") if p.stem.isdigit() else (1 << 30, p.stem)


def iter_refs(cfg: Mapping[str, Any] | None = None, subsets: list[str] | None = None,
              ) -> Iterator[DrawingRef]:
    """Yield every complete drawing in the dataset, in a deterministic order."""
    cfg = cfg or load_config("data")
    root = resolve(cfg["dataset_root"]) / cfg["complete_dir"]
    for key, dirname in cfg["subsets"].items():
        if subsets is not None and key not in subsets:
            continue
        for g in sorted((root / dirname).glob("*.graphml"), key=_sort_key):
            yield DrawingRef(key, g.stem, g, _image_for(g))


def ref_from_uid(uid: str, cfg: Mapping[str, Any] | None = None) -> DrawingRef:
    cfg = cfg or load_config("data")
    subset, drawing_id = uid.split("/", 1)
    g = resolve(cfg["dataset_root"]) / cfg["complete_dir"] / cfg["subsets"][subset]
    g = g / f"{drawing_id}.graphml"
    return DrawingRef(subset, drawing_id, g, _image_for(g))


def graph_to_drawing(G: nx.Graph, drawing_id: str, label_map: Mapping[str, str | None],
                     subset: str = "") -> Drawing:
    symbols: list[Symbol] = []
    ignore: list[BBox] = []
    kept: set[str] = set()
    for nid, d in G.nodes(data=True):
        raw = d.get("label")
        if raw not in label_map:
            raise ValueError(f"{subset}/{drawing_id}: unknown node label {raw!r}")
        bbox = BBox(xmin=float(d["xmin"]), ymin=float(d["ymin"]),
                    xmax=float(d["xmax"]), ymax=float(d["ymax"]))
        cls = label_map[raw]
        if cls is None:
            ignore.append(bbox)
            continue
        symbols.append(Symbol(id=str(nid), cls=cls, bbox=bbox))
        kept.add(str(nid))

    edges: list[Edge] = []
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        u, v = str(u), str(v)
        if u not in kept or v not in kept:
            raise ValueError(f"{subset}/{drawing_id}: edge touches non-graph node {u}-{v}")
        edges.append(Edge(id=f"e{i}", source=u, target=v,
                          line_type=LineType(d.get("edge_label", "solid"))))
    return Drawing(id=drawing_id, subset=subset, symbols=symbols, edges=edges,
                   ignore_regions=ignore)


def load_drawing(ref: DrawingRef, cfg: Mapping[str, Any] | None = None,
                 with_size: bool = True) -> Drawing:
    cfg = cfg or load_config("data")
    G = nx.read_graphml(ref.graphml)
    if G.is_directed() or G.is_multigraph():
        raise ValueError(f"{ref.uid}: expected a simple undirected graph")
    drawing = graph_to_drawing(G, ref.drawing_id, cfg["label_map"], ref.subset)
    drawing.image_path = str(ref.image)
    if with_size:
        from PIL import Image

        with Image.open(ref.image) as im:  # header read only; no full decode
            drawing.width, drawing.height = im.size
    return drawing
