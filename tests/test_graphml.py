from __future__ import annotations

from pathlib import Path

from pid.export.graphml import read_graphml, to_graphml, validate_graphml, write_graphml
from pid.io.drawing import BBox, Drawing

DATA = Path(__file__).parent / "data"


def test_round_trip_is_exact(mini: Drawing, tmp_path: Path) -> None:
    d = mini.model_copy(deep=True)
    d.symbols[1].score = 0.123456789012345  # repr-exact floats
    d.edges[0].score = 0.5
    back = read_graphml(write_graphml(d, tmp_path / "x.graphml"))
    assert back == d
    # GraphML -> Drawing -> GraphML is byte-identical
    assert to_graphml(back) == to_graphml(d)


def test_round_trip_grid(grid: Drawing, tmp_path: Path) -> None:
    assert read_graphml(write_graphml(grid, tmp_path / "g.graphml")) == grid


def test_output_is_xsd_valid(mini: Drawing, grid: Drawing) -> None:
    assert validate_graphml(to_graphml(mini)) == []
    assert validate_graphml(to_graphml(grid)) == []


def test_ground_truth_fixture_is_xsd_valid() -> None:
    assert validate_graphml((DATA / "mini.graphml").read_bytes()) == []


def test_xsd_rejects_malformed() -> None:
    bad = b'<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><node id="a"/></graphml>'
    assert validate_graphml(bad) != []


def test_raw_label_spelling_and_background(mini: Drawing) -> None:
    xml = to_graphml(mini).decode()
    assert ">connector<" in xml and ">inlet/outlet<" in xml and ">background<" in xml
    assert ">ankle<" not in xml


def test_empty_drawing(tmp_path: Path) -> None:
    d = Drawing(id="empty", subset="s", width=10, height=10,
                ignore_regions=[BBox(xmin=0, ymin=0, xmax=1, ymax=1)])
    assert read_graphml(write_graphml(d, tmp_path / "e.graphml")) == d


def test_edge_orientation_and_order_preserved(tmp_path: Path) -> None:
    from pid.io.drawing import Edge, Symbol

    def s(i: str, x: float) -> Symbol:
        return Symbol(id=i, cls="valve", bbox=BBox(xmin=x, ymin=0, xmax=x + 1, ymax=1))

    d = Drawing(id="o", symbols=[s("a", 0), s("b", 5), s("c", 9)],
                edges=[Edge(id="e1", source="c", target="a"),
                       Edge(id="e0", source="b", target="a")])
    assert read_graphml(write_graphml(d, tmp_path / "o.graphml")) == d


def test_invalid_dataset_ids_are_sanitized(data_cfg: dict[str, object]) -> None:
    import networkx as nx

    from pid.io.pid2graph import graph_to_drawing

    G = nx.Graph()
    G.add_node("inlet/outlet3", label="inlet/outlet", xmin=0, ymin=0, xmax=1, ymax=1)
    G.add_node("valve1", label="valve", xmin=2, ymin=0, xmax=3, ymax=1)
    G.add_edge("inlet/outlet3", "valve1", edge_label="solid")
    d = graph_to_drawing(G, "x", data_cfg["label_map"])  # type: ignore[arg-type]
    assert {sym.id for sym in d.symbols} == {"inlet_outlet3", "valve1"}
    assert validate_graphml(to_graphml(d)) == []


def test_incompatible_schema_version_rejected(mini: Drawing, tmp_path: Path) -> None:
    import pytest

    p = write_graphml(mini, tmp_path / "v.graphml")
    p.write_bytes(p.read_bytes().replace(b">1.0<", b">2.0<", 1))
    with pytest.raises(ValueError, match="schema_version"):
        read_graphml(p)
