"""Assemble a graph from symbols and orthogonal line segments.

Follows the PID2Graph annotation convention (DATASET_NOTES.md):

* where a horizontal and a vertical segment meet (bend, T or X) there is one
  `crossing` node, connected to its neighbours along both segments - the GT does not
  distinguish junctions from pass-through crossings;
* where a segment ends at a symbol there is an `ankle` node on the symbol's bbox
  border, connected to the symbol;
* consecutive nodes along a segment are connected by an edge with the segment's
  line type.
Structural nodes are 8x8 boxes centered on their point, as in the dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pid.io.drawing import BBox, Drawing, Edge, LineType, Symbol
from pid.lines.extract import Segment


def merge_collinear(segs: Sequence[Segment], c_tol: float, max_gap: float) -> list[Segment]:
    """Join same-orientation segments on (almost) the same centerline separated by at
    most `max_gap` pixels (dash gaps, small breaks)."""
    out: list[Segment] = []
    for orient in ("h", "v"):
        group = sorted((s for s in segs if s.orient == orient), key=lambda s: (s.c, s.a0))
        # cluster by centerline, then sweep along the axis
        clusters: list[list[Segment]] = []
        for s in group:
            if clusters and abs(s.c - clusters[-1][-1].c) <= c_tol:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        for cl in clusters:
            cl.sort(key=lambda s: s.a0)
            cur = Segment(cl[0].orient, cl[0].c, cl[0].a0, cl[0].a1, cl[0].width, cl[0].fill,
                          cl[0].line_type)
            for s in cl[1:]:
                if s.a0 - cur.a1 <= max_gap:
                    la, lb = max(cur.length, 1.0), max(s.length, 1.0)
                    cur.c = (cur.c * la + s.c * lb) / (la + lb)
                    cur.fill = (cur.fill * la + s.fill * lb) / (la + lb)
                    if s.line_type != cur.line_type:
                        cur.line_type = cur.line_type if la >= lb else s.line_type
                    cur.a1 = max(cur.a1, s.a1)
                    cur.width = max(cur.width, s.width)
                else:
                    out.append(cur)
                    cur = Segment(s.orient, s.c, s.a0, s.a1, s.width, s.fill, s.line_type)
            out.append(cur)
    return out


class _Nodes:
    """Point store that merges points closer than `tol` (L-inf)."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.pts: list[tuple[float, float]] = []
        self.kind: list[str] = []
        self.symbol: list[str | None] = []

    def add(self, x: float, y: float, kind: str, symbol: str | None = None) -> int:
        for i, (px, py) in enumerate(self.pts):
            if abs(px - x) <= self.tol and abs(py - y) <= self.tol:
                if kind == "ankle" and self.kind[i] != "ankle":
                    self.kind[i], self.symbol[i] = kind, symbol
                return i
        self.pts.append((x, y))
        self.kind.append(kind)
        self.symbol.append(symbol)
        return len(self.pts) - 1


def _snap_to_symbol(seg: Segment, end: str, symbols: Sequence[Symbol], tol: float,
                    ) -> tuple[Symbol, tuple[float, float]] | None:
    """Find the symbol a segment end runs into; return it and the ankle point on its
    border (where the line enters the box)."""
    a = seg.a1 if end == "hi" else seg.a0
    best: tuple[float, Symbol, tuple[float, float]] | None = None
    for s in symbols:
        b = s.bbox
        if seg.orient == "h":
            if not (b.ymin - tol <= seg.c <= b.ymax + tol):
                continue
            edge = b.xmin if end == "hi" else b.xmax  # the side the line enters through
            d = abs(edge - a) if (b.xmin - tol <= a <= b.xmax + tol) else np.inf
            pt = (edge, min(max(seg.c, b.ymin), b.ymax))
        else:
            if not (b.xmin - tol <= seg.c <= b.xmax + tol):
                continue
            edge = b.ymin if end == "hi" else b.ymax
            d = abs(edge - a) if (b.ymin - tol <= a <= b.ymax + tol) else np.inf
            pt = (min(max(seg.c, b.xmin), b.xmax), edge)
        if d <= tol and (best is None or d < best[0]):
            best = (d, s, pt)
    return None if best is None else (best[1], best[2])


def _drop_inactive_crossings(members: list[list[tuple[float, int]]], nodes: _Nodes,
                             segs: Sequence[Segment], join_tol: float) -> None:
    """Remove crossing nodes that only one segment actually uses.

    A segment "uses" a node if it has at least one other node to connect it to. A
    crossing used by a single segment and lying strictly inside it would split a
    straight run into two edges - the dataset never has a degree-2 node on a straight
    line - so it is a spur (text stroke, symbol remnant) touching the line. Iterate,
    because removing a node can leave another segment with a single member.
    """
    while True:
        counts: dict[int, list[int]] = {}
        for k, mem in enumerate(members):
            distinct = {n for _, n in mem}
            if len(distinct) >= 2:
                for n in distinct:
                    counts.setdefault(n, []).append(k)
        drop: set[tuple[int, int]] = set()
        for k, mem in enumerate(members):
            for a, n in mem:
                if nodes.kind[n] != "crossing":
                    continue
                used = counts.get(n, [])
                interior = segs[k].a0 + join_tol < a < segs[k].a1 - join_tol
                if len(used) == 1 and used[0] == k and interior:
                    drop.add((k, n))
        if not drop:
            return
        for k, n in drop:
            members[k] = [(a, m) for a, m in members[k] if m != n]


def _prune_symbolless(edge_set: dict[frozenset[str], LineType], symbol_ids: set[str],
                      ) -> dict[frozenset[str], LineType]:
    """Drop connected components that contain no symbol: text strokes, table rules
    in the title block and the drawing frame form line networks with no equipment."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in edge_set:
        u, v = tuple(key)
        parent[find(u)] = find(v)
    keep_roots = {find(n) for n in parent if n in symbol_ids}
    return {k: lt for k, lt in edge_set.items() if find(next(iter(k))) in keep_roots}


def assemble(drawing_id: str, symbols: Sequence[Symbol], segments: Sequence[Segment],
             cfg: Mapping[str, Any], subset: str = "") -> Drawing:
    join_tol = float(cfg["join_tol"])
    snap_tol = float(cfg["snap_tol"])
    half = float(cfg.get("struct_size", 8)) / 2
    segs = merge_collinear(segments, float(cfg["collinear_tol"]), float(cfg["merge_gap"]))
    segs = [s for s in segs if s.length >= float(cfg.get("min_segment_length", 0))]
    nodes = _Nodes(float(cfg.get("node_merge_tol", 3)))
    members: list[list[tuple[float, int]]] = [[] for _ in segs]  # (along, node) per segment

    H = [i for i, s in enumerate(segs) if s.orient == "h"]
    V = [i for i, s in enumerate(segs) if s.orient == "v"]
    if H and V:
        hc = np.array([segs[i].c for i in H])
        ha0 = np.array([segs[i].a0 for i in H])
        ha1 = np.array([segs[i].a1 for i in H])
        vc = np.array([segs[i].c for i in V])
        va0 = np.array([segs[i].a0 for i in V])
        va1 = np.array([segs[i].a1 for i in V])
        hit = ((vc[None, :] >= ha0[:, None] - join_tol) & (vc[None, :] <= ha1[:, None] + join_tol)
               & (hc[:, None] >= va0[None, :] - join_tol)
               & (hc[:, None] <= va1[None, :] + join_tol))
        for hi, vi in zip(*np.nonzero(hit), strict=True):
            ih, iv = H[hi], V[vi]
            x, y = segs[iv].c, segs[ih].c
            n = nodes.add(x, y, "crossing")
            members[ih].append((x, n))
            members[iv].append((y, n))

    sym_edges: set[tuple[int, str]] = set()
    for k, s in enumerate(segs):
        for end in ("lo", "hi"):
            a = s.a0 if end == "lo" else s.a1
            if any(abs(m_a - a) <= join_tol for m_a, _ in members[k]):
                continue  # this end is already a junction
            snap = _snap_to_symbol(s, end, symbols, snap_tol)
            if snap is None:
                continue  # dangling end: no node (walking skeleton)
            sym, (px, py) = snap
            n = nodes.add(px, py, "ankle", sym.id)
            members[k].append((py if s.orient == "v" else px, n))
            sym_edges.add((n, sym.id))

    _drop_inactive_crossings(members, nodes, segs, join_tol)

    edge_set: dict[frozenset[str], LineType] = {}

    def nid(i: int) -> str:
        return f"{nodes.kind[i][0]}{i}"

    for k, s in enumerate(segs):
        seq = sorted(set(members[k]))
        for (_, u), (_, v) in zip(seq, seq[1:], strict=False):
            if u != v:
                edge_set.setdefault(frozenset((nid(u), nid(v))), s.line_type)
    for n, sid in sym_edges:
        edge_set.setdefault(frozenset((nid(n), sid)), LineType.SOLID)

    if cfg.get("prune_symbolless_components", True):
        edge_set = _prune_symbolless(edge_set, {s.id for s in symbols})
    used = {x for key in edge_set for x in key}
    out_syms = list(symbols)
    for i, (x, y) in enumerate(nodes.pts):
        if nid(i) in used:
            out_syms.append(Symbol(id=nid(i), cls=nodes.kind[i],
                                   bbox=BBox(xmin=x - half, ymin=y - half, xmax=x + half,
                                             ymax=y + half)))
    edges = []
    for j, (key, lt) in enumerate(sorted(edge_set.items(), key=lambda t: sorted(t[0]))):
        src, dst = sorted(key)
        edges.append(Edge(id=f"e{j}", source=src, target=dst, line_type=lt))
    return Drawing(id=drawing_id, subset=subset, symbols=out_syms, edges=edges)
