"""DEXPI-structured export: structure, mapping coverage and validation."""

from __future__ import annotations

from lxml import etree

from pid.export.dexpi import load_mapping, to_dexpi, validate_dexpi
from pid.io.drawing import BBox, Drawing, Edge, LineType, Symbol


def s(i: str, cls: str, x: float, y: float, w: float = 8) -> Symbol:
    return Symbol(id=i, cls=cls, bbox=BBox(xmin=x, ymin=y, xmax=x + w, ymax=y + w))


def plant() -> Drawing:
    """Small plant: tank -> valve -> bend -> T-junction, whose branches go to a pump
    and to an unmapped `general` symbol; an instrument taps the junction with a dashed
    signal line."""
    syms = [s("tank", "tank", 0, 0, 100), s("v", "valve", 300, 40, 30), s("p", "pump", 700, 40, 60),
            s("g", "general", 500, 300, 40), s("i", "instrumentation", 500, -200, 40),
            s("a1", "ankle", 100, 50), s("a2", "ankle", 300, 50), s("a3", "ankle", 330, 50),
            s("bend", "crossing", 400, 50), s("tee", "crossing", 500, 50),
            s("a4", "ankle", 700, 50), s("a5", "ankle", 500, 300), s("a6", "ankle", 500, -160)]
    pairs = [("tank", "a1"), ("a1", "a2"), ("a2", "v"), ("v", "a3"), ("a3", "bend"),
             ("bend", "tee"), ("tee", "a4"), ("a4", "p"), ("tee", "a5"), ("a5", "g"),
             ("i", "a6"), ("a6", "tee")]
    edges = [Edge(id=f"e{k}", source=u, target=v,
                  line_type=LineType.NON_SOLID if u == "a6" else LineType.SOLID)
             for k, (u, v) in enumerate(pairs)]
    return Drawing(id="fx", subset="fixture", symbols=syms, edges=edges)


def types(xml: bytes) -> list[str]:
    return [o.get("type", "") for o in etree.fromstring(xml).iter("Object")]


def test_fixture_exports_valid_dexpi() -> None:
    xml, rep = to_dexpi(plant())
    assert validate_dexpi(xml) == []
    t = types(xml)
    assert t.count("Plant/ProcessEquipment.Tank") == 1
    assert t.count("Plant/ProcessEquipment.Pump") == 1
    assert t.count("Plant/Piping.OperatedValve") == 1
    assert t.count("Plant/Piping.PipeTee") == 1           # degree-3 junction
    assert t.count("Plant/ProcessEquipment.ProcessNozzle") == 2  # tank + pump ports
    assert t.count("Plant/Instrumentation.ProcessSignalGeneratingFunction") == 1
    assert t.count("Plant/Instrumentation.MeasuringLineFunction") == 1
    assert rep.unmapped_symbols == {"general": 1}
    # chains: tank-valve, valve-tee (through the bend), tee-pump, tee-general(open)
    assert rep.chains["piping"] == 4 and rep.chains["piping_one_end_unmapped"] == 1
    assert rep.chains["instrument_piping"] == 1


def test_valve_is_segment_item_with_two_nodes() -> None:
    xml, _ = to_dexpi(plant())
    root = etree.fromstring(xml)
    valve = root.xpath('//Object[@type="Plant/Piping.OperatedValve"]')[0]
    assert valve.getparent().get("property") == "Items"
    assert len(valve.xpath('Components[@property="Nodes"]/Object')) == 2


def test_validator_rejects_bad_documents() -> None:
    xml, _ = to_dexpi(plant())
    root = etree.fromstring(xml)
    tank = root.xpath('//Object[@type="Plant/ProcessEquipment.Tank"]')[0]
    tank.set("type", "Plant/ProcessEquipment.NoSuchThing")
    seg = root.xpath('//Object[@type="Plant/Piping.PipingNetworkSegment"]')[0]
    etree.SubElement(seg, "References", property="Bogus", objects="#missing")
    errs = validate_dexpi(etree.tostring(root))
    assert any("unknown class" in e for e in errs)
    assert any("no property 'Bogus'" in e for e in errs)
    assert any("dangling reference #missing" in e for e in errs)


def test_abstract_class_rejected() -> None:
    xml, _ = to_dexpi(plant())
    root = etree.fromstring(xml)
    root.xpath('//Object[@type="Plant/Piping.PipeTee"]')[0].set("type",
                                                                 "Plant/Piping.PipingComponent")
    assert any("abstract class" in e for e in validate_dexpi(etree.tostring(root)))


def test_mapping_classes_exist_in_vocabulary() -> None:
    from pid.export.dexpi import _vocab

    v, m = _vocab(), load_mapping()
    named = [spec["type"] for spec in m["symbols"].values()]
    st = m["structural"]
    named += [st["ankle_on_equipment"], st["ankle_on_piping_component"], st["junction"],
              st["pipe"], st["segment"], st["system"], st["instrument_to_instrument"],
              *st["instrument_to_piping"].values()]
    for t in named:
        assert t in v and not v[t]["abstract"], t
    assert set(m["symbols"]) | set(m["unmapped"]) == {
        "general", "tank", "valve", "instrumentation", "pump", "arrow", "inlet_outlet"}


def test_empty_drawing() -> None:
    xml, rep = to_dexpi(Drawing(id="e", subset="s"))
    assert validate_dexpi(xml) == []
    assert rep.exported["segment"] == 0


def test_ids_unique_when_node_ids_look_like_chain_ids() -> None:
    """Regression: a crossing named 'c1' must not collide with chain-derived ids."""
    d = plant()
    rename = {"tee": "c1", "bend": "c0"}
    for sy in d.symbols:
        sy.id = rename.get(sy.id, sy.id)
    for e in d.edges:
        e.source, e.target = rename.get(e.source, e.source), rename.get(e.target, e.target)
    xml, _ = to_dexpi(d)
    ids = [o.get("id") for o in etree.fromstring(xml).iter("Object") if o.get("id")]
    assert len(ids) == len(set(ids))
    assert validate_dexpi(xml) == []


def test_loop_chain_back_to_same_tee() -> None:
    """A pipe leaving a junction and returning to it (loop) gives the tee two nodes."""
    syms = [s("t", "tank", 0, 0, 50), s("a", "ankle", 50, 20),
            s("j", "crossing", 200, 20), s("b1", "crossing", 200, 200),
            s("b2", "crossing", 400, 200), s("b3", "crossing", 400, 20)]
    pairs = [("t", "a"), ("a", "j"), ("j", "b1"), ("b1", "b2"), ("b2", "b3"), ("b3", "j")]
    d = Drawing(id="loop", symbols=syms,
                edges=[Edge(id=f"e{k}", source=u, target=v) for k, (u, v) in enumerate(pairs)])
    xml, rep = to_dexpi(d)
    assert validate_dexpi(xml) == []
    tee = etree.fromstring(xml).xpath('//Object[@type="Plant/Piping.PipeTee"]')[0]
    assert len(tee.xpath('Components[@property="Nodes"]/Object')) == 3
