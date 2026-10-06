from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pid.io.drawing import BBox, Drawing, Edge, LineType, Symbol
from pid.io.overlay import render_overlay


def test_mini_parses_by_attr_name(mini: Drawing) -> None:
    by_id = mini.symbol_by_id()
    assert set(by_id) == {"general1", "valve2", "connector3", "crossing4",
                          "instrumentation5", "inlet6"}
    assert by_id["valve2"].bbox.as_xyxy() == (100.5, 15.0, 120.5, 35.0)
    assert by_id["connector3"].bbox.as_xyxy() == (196, 21, 204, 29)  # long-typed keys


def test_label_map(mini: Drawing) -> None:
    cls = {s.id: s.cls for s in mini.symbols}
    assert cls["connector3"] == "ankle"
    assert cls["inlet6"] == "inlet_outlet"


def test_background_becomes_ignore_region_with_fixed_bbox(mini: Drawing) -> None:
    assert len(mini.ignore_regions) == 1
    r = mini.ignore_regions[0]
    assert (r.ymin, r.ymax) == (220.0, 240.0)


def test_edges(mini: Drawing) -> None:
    assert len(mini.edges) == 5
    types = sorted(e.line_type.value for e in mini.edges)
    assert types == ["non-solid"] + ["solid"] * 4


def test_dangling_edge_rejected() -> None:
    s = Symbol(id="a", cls="valve", bbox=BBox(xmin=0, ymin=0, xmax=1, ymax=1))
    with pytest.raises(ValueError):
        Drawing(id="x", symbols=[s], edges=[Edge(id="e", source="a", target="b")])


def test_json_round_trip(mini: Drawing) -> None:
    assert Drawing.model_validate_json(mini.model_dump_json()) == mini


def test_overlay_renders(mini: Drawing, tmp_path: Path) -> None:
    img = Image.new("RGB", (320, 240), "white")
    out = render_overlay(mini, image=img)
    assert out.size == (320, 240)
    # an edge pixel between general1 and valve2 centers should be colored
    assert out.getpixel((80, 25)) != (255, 255, 255)
    assert LineType.SOLID in {e.line_type for e in mini.edges}


def test_overlay_confidence_mode(mini: Drawing) -> None:
    from pid.io.overlay import confidence_color

    d = mini.model_copy(deep=True)
    for e in d.edges:
        e.score = 0.1
    img = Image.new("RGB", (320, 240), "white")
    out = render_overlay(d, image=img, color_by="confidence")
    r, g, _ = out.getpixel((80, 25))
    assert r > g  # low confidence edges render red-ish
    assert confidence_color(1.0)[1] > confidence_color(1.0)[0]
    with pytest.raises(ValueError):
        render_overlay(d, image=img, color_by="nope")
