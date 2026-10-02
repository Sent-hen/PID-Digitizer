from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from pid.config import load_config
from pid.io.drawing import BBox, Drawing, Edge, LineType, Symbol
from pid.io.pid2graph import graph_to_drawing

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_cfg() -> dict[str, Any]:
    return load_config("data")


@pytest.fixture
def mini(data_cfg: dict[str, Any]) -> Drawing:
    d = graph_to_drawing(nx.read_graphml(DATA / "mini.graphml"), "mini",
                         data_cfg["label_map"], "fixture")
    d.width, d.height = 320, 240
    return d


def make_grid_drawing(n: int = 6, seed_cls: tuple[str, ...] = ("general", "valve", "ankle"),
                      ) -> Drawing:
    """A deterministic n x n lattice: every node connected to its right/down neighbor.
    Every third edge is non-solid. Used for oracle tests with known edge counts."""
    symbols, edges = [], []
    for r in range(n):
        for c in range(n):
            cls = seed_cls[(r * n + c) % len(seed_cls)]
            size = 8 if cls == "ankle" else 30
            x, y = 100 * c + 20, 100 * r + 20
            symbols.append(Symbol(id=f"n{r}_{c}", cls=cls,
                                  bbox=BBox(xmin=x, ymin=y, xmax=x + size, ymax=y + size)))
    for r in range(n):
        for c in range(n):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < n and cc < n:
                    lt = LineType.NON_SOLID if len(edges) % 3 == 0 else LineType.SOLID
                    edges.append(Edge(id=f"e{len(edges)}", source=f"n{r}_{c}",
                                      target=f"n{rr}_{cc}", line_type=lt))
    return Drawing(id="grid", subset="fixture", width=100 * n, height=100 * n,
                   symbols=symbols, edges=edges)


@pytest.fixture
def grid() -> Drawing:
    return make_grid_drawing()
