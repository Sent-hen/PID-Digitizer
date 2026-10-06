"""GraphML export/import - the project's primary output format (proposal v3 §4).

Uses the PID2Graph ground-truth attribute names (`label`, `xmin`..`ymax`,
`edge_label`) and raw label spellings, so a prediction file is read by the same
loader as ground truth. Adds `confidence` on nodes/edges and drawing metadata on the
graph. Ignore regions are written as `background` nodes, as in the dataset.
Floats are written with `repr`, so Drawing -> GraphML -> Drawing is exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import networkx as nx
from lxml import etree

from pid.config import load_config
from pid.io.drawing import BBox, Drawing
from pid.io.pid2graph import graph_to_drawing, safe_id

SCHEMA_VERSION = "1.0"
NS = "http://graphml.graphdrawing.org/xmlns"
XSD_DIR = Path(__file__).parent / "xsd"

# (id, for, attr.name, attr.type)
KEYS: list[tuple[str, str, str, str]] = [
    ("g_schema", "graph", "schema_version", "string"),
    ("g_drawing", "graph", "drawing_id", "string"),
    ("g_subset", "graph", "subset", "string"),
    ("g_source", "graph", "source_file", "string"),
    ("g_width", "graph", "width", "int"),
    ("g_height", "graph", "height", "int"),
    ("g_pipeline", "graph", "pipeline_version", "string"),
    ("g_config", "graph", "config_hash", "string"),
    ("label", "node", "label", "string"),
    ("xmin", "node", "xmin", "double"),
    ("ymin", "node", "ymin", "double"),
    ("xmax", "node", "xmax", "double"),
    ("ymax", "node", "ymax", "double"),
    ("n_conf", "node", "confidence", "double"),
    ("edge_label", "edge", "edge_label", "string"),
    ("e_conf", "edge", "confidence", "double"),
]


def _raw_labels(label_map: Mapping[str, str | None]) -> dict[str, str]:
    """canonical class -> raw dataset spelling (inverse of data.yaml label_map)."""
    return {v: k for k, v in label_map.items() if v is not None}


def to_graphml(d: Drawing, label_map: Mapping[str, str | None] | None = None,
               pipeline_version: str = "", config_hash: str = "") -> bytes:
    label_map = label_map or load_config("data")["label_map"]
    raw = _raw_labels(label_map)
    bg = next(k for k, v in label_map.items() if v is None)
    root = etree.Element(f"{{{NS}}}graphml", nsmap={None: NS})
    for kid, kfor, name, typ in KEYS:
        etree.SubElement(root, f"{{{NS}}}key", id=kid, attrib={
            "for": kfor, "attr.name": name, "attr.type": typ})
    g = etree.SubElement(root, f"{{{NS}}}graph", id=d.id or "G", edgedefault="undirected")

    def data(parent: etree._Element, key: str, value: Any) -> None:
        el = etree.SubElement(parent, f"{{{NS}}}data", key=key)
        el.text = repr(value) if isinstance(value, float) else str(value)

    meta: list[tuple[str, Any]] = [("g_schema", SCHEMA_VERSION), ("g_drawing", d.id),
                                   ("g_subset", d.subset)]
    if d.image_path:
        meta.append(("g_source", Path(d.image_path).name))
    if d.width is not None and d.height is not None:
        meta += [("g_width", d.width), ("g_height", d.height)]
    if pipeline_version:
        meta.append(("g_pipeline", pipeline_version))
    if config_hash:
        meta.append(("g_config", config_hash))
    for k, v in meta:
        data(g, k, v)

    def bbox(parent: etree._Element, b: BBox) -> None:
        for k in ("xmin", "ymin", "xmax", "ymax"):
            data(parent, k, float(getattr(b, k)))

    for s in d.symbols:
        n = etree.SubElement(g, f"{{{NS}}}node", id=s.id)
        data(n, "label", raw.get(s.cls, s.cls))
        bbox(n, s.bbox)
        data(n, "n_conf", float(s.score))
    for i, r in enumerate(d.ignore_regions):
        n = etree.SubElement(g, f"{{{NS}}}node", id=f"background{i}")
        data(n, "label", bg)
        bbox(n, r)
    for e in d.edges:
        el = etree.SubElement(g, f"{{{NS}}}edge", id=e.id, source=e.source, target=e.target)
        data(el, "edge_label", e.line_type.value)
        data(el, "e_conf", float(e.score))
    out: bytes = etree.tostring(root, xml_declaration=True, encoding="utf-8",
                                pretty_print=True)
    return out


def write_graphml(d: Drawing, path: Path, **kw: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_graphml(d, **kw))
    return path


def read_graphml(path: Path, label_map: Mapping[str, str | None] | None = None,
                 ) -> Drawing:
    """Read a prediction (or ground-truth) GraphML file into a Drawing."""
    label_map = label_map or load_config("data")["label_map"]
    G = nx.read_graphml(path)
    gd = G.graph
    d = graph_to_drawing(G, str(gd.get("drawing_id", path.stem)), label_map,
                         str(gd.get("subset", "")))
    # networkx returns undirected edges in adjacency order with arbitrary endpoint
    # orientation; restore both from the file so the round trip is exact.
    file_edges = {el.get("id"): (i, el.get("source"), el.get("target")) for i, el in
                  enumerate(etree.parse(str(path)).iter(f"{{{NS}}}edge"))}
    for e in d.edges:
        if e.id in file_edges:
            _, src, dst = file_edges[e.id]
            e.source, e.target = safe_id(src), safe_id(dst)
    d.edges.sort(key=lambda e: file_edges.get(e.id, (len(file_edges),))[0])
    d.width = int(gd["width"]) if "width" in gd else None
    d.height = int(gd["height"]) if "height" in gd else None
    return d


@cache
def _schema() -> etree.XMLSchema:
    return etree.XMLSchema(etree.parse(str(XSD_DIR / "graphml.xsd")))


def validate_graphml(data: bytes) -> list[str]:
    """Validate against the official GraphML 1.0 XSD; returns error messages."""
    schema = _schema()
    ok = schema.validate(etree.fromstring(data))
    return [] if ok else [str(err) for err in schema.error_log]
