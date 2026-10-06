"""DEXPI-structured XML export (DEXPI 2.0.1 "DEXPI XML" serialization).

Topology-only export of a `Drawing` (no diagram geometry, tags or line numbers -
the pipeline does not produce them). Classes come from `dexpi_mapping.yaml`;
unmapped classes are reported, never guessed. Output is validated against the
official DEXPI XML Schema and against the DEXPI model vocabulary (class exists and
is concrete; every Components/References property belongs to the class or one of its
supertypes; every reference resolves). It is called "DEXPI-structured" because it has
not been imported into a DEXPI-certified tool (DECISIONS D-023).

Translation (see the mapping file):
* ankle on a tank/pump -> ProcessNozzle (with a PipingNode) of that equipment;
  ankle on a valve -> PipingNode of the OperatedValve;
* crossing with >= 3 pipe branches -> PipeTee (one PipingNode per branch);
  degree-2 crossings (bends) are folded into the pipe;
* every chain of edges between two terminals -> one PipingNetworkSegment holding one
  Pipe connection; segments are grouped into one PipingNetworkSystem per connected
  piping network;
* chain between an instrument and a piping item -> ProcessSignalGeneratingFunction
  (SensingLocation = the piping item) + MeasuringLineFunction to the instrument;
  instrument-instrument chains -> SignalConveyingFunction.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from lxml import etree

from pid.io.drawing import Drawing

HERE = Path(__file__).parent
XSD = HERE / "dexpi_xsd" / "DEXPI_XML_Schema.xsd"
VOCAB = HERE / "dexpi_xsd" / "dexpi_vocabulary.json"
MAPPING = HERE / "dexpi_mapping.yaml"
STRUCTURAL = ("crossing", "ankle")
_ID_BAD = re.compile(r"[^A-Za-z0-9_]")


@cache
def load_mapping() -> dict[str, Any]:
    data = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _sid(prefix: str, raw: str) -> str:
    return f"{prefix}_{_ID_BAD.sub('_', raw)}"


@dataclass
class _Terminal:
    """End of a pipe chain: an item (object id) and the node on it (or None)."""
    kind: str               # equipment | piping_component | instrument | tee | open
    item: str | None        # DEXPI object id of the item
    node: str | None        # DEXPI PipingNode id (None for instruments / open ends)
    symbol: str | None = None


@dataclass
class DexpiReport:
    exported: Counter[str] = field(default_factory=Counter)
    unmapped_symbols: Counter[str] = field(default_factory=Counter)
    chains: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {"exported": dict(self.exported), "unmapped_symbols": dict(self.unmapped_symbols),
                "chains": dict(self.chains)}


class _Builder:
    def __init__(self, d: Drawing, mapping: Mapping[str, Any]) -> None:
        self.d, self.m = d, mapping
        self.report = DexpiReport()
        self.sym = {s.id: s for s in d.symbols}
        self.adj: dict[str, list[tuple[str, int]]] = defaultdict(list)  # node -> (nbr, edge#)
        for k, e in enumerate(d.edges):
            self.adj[e.source].append((e.target, k))
            self.adj[e.target].append((e.source, k))
        self.kind = {sid: self._kind(s.cls) for sid, s in self.sym.items()}
        # DEXPI objects being built
        self.equipment: dict[str, etree._Element] = {}
        self.items: dict[str, etree._Element] = {}       # piping components and tees
        self.instruments: dict[str, etree._Element] = {}
        self.placed: set[str] = set()                    # items already put in a segment
        self.ports: dict[str, _Terminal] = {}            # ankle id -> terminal

    def _kind(self, cls: str) -> str:
        if cls in STRUCTURAL:
            return cls
        spec = self.m["symbols"].get(cls)
        if spec is None:
            return "unmapped"
        return str(spec["kind"])

    # --- XML helpers -----------------------------------------------------------
    @staticmethod
    def obj(type_: str, id_: str | None = None) -> etree._Element:
        el = etree.Element("Object", type=type_)
        if id_:
            el.set("id", id_)
        return el

    @staticmethod
    def comp(parent: etree._Element, prop: str) -> etree._Element:
        for c in parent.findall("Components"):
            if c.get("property") == prop:
                return c
        return etree.SubElement(parent, "Components", property=prop)

    @staticmethod
    def ref(parent: etree._Element, prop: str, *ids: str) -> None:
        etree.SubElement(parent, "References", property=prop,
                         objects=" ".join(f"#{i}" for i in ids))

    def new_node(self, owner: etree._Element, raw: str) -> str:
        nid = _sid("Node", raw)
        self.comp(owner, "Nodes").append(self.obj("Plant/Piping.PipingNode", nid))
        return nid

    # --- symbols and ports -----------------------------------------------------------
    def build_symbols(self) -> None:
        for sid, s in self.sym.items():
            k = self.kind[sid]
            if k == "unmapped":
                self.report.unmapped_symbols[s.cls] += 1
            elif k == "equipment":
                self.equipment[sid] = self.obj(self.m["symbols"][s.cls]["type"], _sid("Eq", sid))
            elif k == "piping_component":
                self.items[sid] = self.obj(self.m["symbols"][s.cls]["type"], _sid("Vl", sid))
            elif k == "instrument":
                self.instruments[sid] = self.obj(self.m["symbols"][s.cls]["type"],
                                                 _sid("PIF", sid))
            if k not in STRUCTURAL and k != "unmapped":
                self.report.exported[s.cls] += 1

    def port_for(self, ankle: str) -> _Terminal:
        """Terminal for an ankle: the (first) symbol it is attached to."""
        if ankle in self.ports:
            return self.ports[ankle]
        owners = [n for n, _ in self.adj[ankle] if self.kind.get(n) not in STRUCTURAL]
        t = _Terminal("open", None, None)
        for sid in owners:
            k = self.kind[sid]
            if k == "equipment":
                eq = self.equipment[sid]
                nz = self.obj(self.m["structural"]["ankle_on_equipment"], _sid("Nz", ankle))
                self.comp(eq, "Nozzles").append(nz)
                t = _Terminal(k, nz.get("id"), self.new_node(nz, ankle), sid)
                self.report.exported["nozzle"] += 1
                break
            if k == "piping_component":
                t = _Terminal(k, self.items[sid].get("id"), self.new_node(self.items[sid], ankle),
                              sid)
                break
            if k == "instrument":
                t = _Terminal(k, self.instruments[sid].get("id"), None, sid)
                break
        self.ports[ankle] = t
        return t

    # --- chains -----------------------------------------------------------------------
    def pipe_degree(self, n: str) -> int:
        return sum(1 for x, _ in self.adj[n] if self.kind.get(x) in STRUCTURAL)

    def is_terminal(self, n: str) -> bool:
        k = self.kind[n]
        if k == "ankle":
            return True
        return k == "crossing" and self.pipe_degree(n) != 2

    def chains(self) -> list[tuple[str, str, list[int]]]:
        """Maximal chains of structural-structural edges between terminal nodes."""
        seen: set[int] = set()
        out = []
        for start in self.sym:
            if self.kind[start] not in STRUCTURAL or not self.is_terminal(start):
                continue
            for nbr, k in self.adj[start]:
                if k in seen or self.kind.get(nbr) not in STRUCTURAL:
                    continue
                path, cur = [k], nbr
                seen.add(k)
                while not self.is_terminal(cur):
                    nxt = [(x, kk) for x, kk in self.adj[cur]
                           if kk not in seen and self.kind.get(x) in STRUCTURAL]
                    if not nxt:
                        break
                    x, kk = nxt[0]
                    seen.add(kk)
                    path.append(kk)
                    cur = x
                out.append((start, cur, path))
        # direct symbol-symbol edges (OPEN100): one chain with implicit ports
        for k, e in enumerate(self.d.edges):
            if self.kind[e.source] not in STRUCTURAL and self.kind[e.target] not in STRUCTURAL:
                out.append((e.source, e.target, [k]))
        return out

    def terminal(self, n: str, chain_key: str) -> _Terminal:
        k = self.kind[n]
        if k == "ankle":
            return self.port_for(n)
        if k == "crossing":
            if self.pipe_degree(n) >= 3:
                if n not in self.items:
                    self.items[n] = self.obj(self.m["structural"]["junction"], _sid("Tee", n))
                    self.report.exported["tee"] += 1
                return _Terminal("tee", self.items[n].get("id"),
                                 self.new_node(self.items[n], f"{n}_{chain_key}"), n)
            return _Terminal("open", None, None)
        # direct symbol end (no ankle): treat like a port on that symbol
        if k == "equipment":
            nz = self.obj(self.m["structural"]["ankle_on_equipment"],
                          _sid("Nz", f"{n}_{chain_key}"))
            self.comp(self.equipment[n], "Nozzles").append(nz)
            return _Terminal(k, nz.get("id"), self.new_node(nz, f"{n}_{chain_key}"), n)
        if k == "piping_component":
            return _Terminal(k, self.items[n].get("id"),
                             self.new_node(self.items[n], f"{n}_{chain_key}"), n)
        if k == "instrument":
            return _Terminal(k, self.instruments[n].get("id"), None, n)
        return _Terminal("open", None, None)

    def build(self) -> etree._Element:
        m = self.m
        self.build_symbols()
        segments: list[tuple[etree._Element, set[str]]] = []  # (segment, item ids it links)
        psgf_count = 0
        for ci, (a, b, path) in enumerate(self.chains()):
            ta, tb = self.terminal(a, f"ch{ci}a"), self.terminal(b, f"ch{ci}b")
            kinds = {ta.kind, tb.kind}
            if "instrument" in kinds:
                inst, other = (ta, tb) if ta.kind == "instrument" else (tb, ta)
                assert inst.symbol is not None
                pif = self.instruments[inst.symbol]
                if other.kind == "instrument":
                    scf = self.obj(m["structural"]["instrument_to_instrument"], f"SCF_ch{ci}")
                    self.ref(scf, "Source", str(inst.item))
                    self.ref(scf, "Target", str(other.item))
                    self.comp(pif, "SignalConveyingFunctions").append(scf)
                    self.report.chains["instrument_instrument"] += 1
                elif other.item is not None:
                    psgf_count += 1
                    g = self.obj(m["structural"]["instrument_to_piping"]["generator"],
                                 f"PSGF_ch{ci}")
                    self.ref(g, "SensingLocation", other.item)
                    self.comp(pif, "ProcessSignalGeneratingFunctions").append(g)
                    line = self.obj(m["structural"]["instrument_to_piping"]["line"], f"MLF_ch{ci}")
                    self.ref(line, "Source", str(g.get("id")))
                    self.ref(line, "Target", str(inst.item))
                    self.comp(pif, "SignalConveyingFunctions").append(line)
                    self.report.chains["instrument_piping"] += 1
                else:
                    self.report.chains["instrument_open_dropped"] += 1
                continue
            if ta.item is None and tb.item is None:
                self.report.chains["both_ends_unmapped_dropped"] += 1
                continue
            seg = self.obj(m["structural"]["segment"], f"Seg_ch{ci}")
            pipe = self.obj(m["structural"]["pipe"], f"Pipe_ch{ci}")
            linked = set()
            for t, end in ((ta, "Source"), (tb, "Target")):
                if t.item is None:
                    continue
                linked.add(t.item)
                for el in (pipe, seg):
                    self.ref(el, f"{end}Item", t.item)
                    if t.node:
                        self.ref(el, f"{end}Node", t.node)
                if t.kind in ("piping_component", "tee") and t.item not in self.placed:
                    sym = t.symbol
                    assert sym is not None
                    self.comp(seg, "Items").append(self.items[sym])
                    self.placed.add(t.item)
            self.comp(seg, "Connections").append(pipe)
            segments.append((seg, linked))
            nonsolid = sum(self.d.edges[k].line_type.value == "non-solid" for k in path)
            self.report.chains["piping_nonsolid" if 2 * nonsolid > len(path) else "piping"] += 1
            if ta.item is None or tb.item is None:
                self.report.chains["piping_one_end_unmapped"] += 1
        self.report.exported["process_signal_generating_function"] += psgf_count

        # Items never linked by a segment (e.g. a valve whose ports all go to
        # instruments or unmapped symbols) still need an owner: give each its own
        # segment so it is exported and SensingLocation references resolve.
        for sid, item in self.items.items():
            if item.get("id") not in self.placed:
                seg = self.obj(m["structural"]["segment"], _sid("SegItem", sid))
                self.comp(seg, "Items").append(item)
                segments.append((seg, {str(item.get("id"))}))
                self.placed.add(str(item.get("id")))

        # One PipingNetworkSystem per connected group of segments (shared items).
        parent = list(range(len(segments)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        owner: dict[str, int] = {}
        for i, (_, linked) in enumerate(segments):
            for it in linked:
                if it in owner:
                    parent[find(i)] = find(owner[it])
                else:
                    owner[it] = i
        groups: dict[int, list[etree._Element]] = defaultdict(list)
        for i, (seg, _) in enumerate(segments):
            groups[find(i)].append(seg)

        plant = self.obj("Plant/PlantModel", "PlantModel1")
        for eq in self.equipment.values():
            self.comp(plant, "TaggedPlantItems").append(eq)
        for pif in self.instruments.values():
            self.comp(plant, "ProcessInstrumentationFunctions").append(pif)
        for gi, segs in enumerate(groups.values()):
            system = self.obj(m["structural"]["system"], f"System{gi + 1}")
            for seg in segs:
                self.comp(system, "Segments").append(seg)
            self.comp(plant, "PipingNetworkSystems").append(system)
        self.report.exported["segment"] = len(segments)
        self.report.exported["system"] = len(groups)
        return plant


def to_dexpi(d: Drawing, mapping: Mapping[str, Any] | None = None,
             ) -> tuple[bytes, DexpiReport]:
    mapping = mapping or load_mapping()
    b = _Builder(d, mapping)
    plant = b.build()
    root = etree.Element("Model", name=_ID_BAD.sub("_", f"drawing_{d.subset}_{d.id}"),
                         uri=f"urn:pid-digitizer:{d.subset or 'drawing'}:{d.id}")
    for prefix, src in mapping["imports"].items():
        etree.SubElement(root, "Import", prefix=prefix, source=src)
    em = etree.SubElement(root, "Object", type="Core/EngineeringModel")
    data = etree.SubElement(em, "Data", property="OriginatingSystemName")
    etree.SubElement(data, "String").text = "pid-digitizer"
    etree.SubElement(em, "Components", property="ConceptualModel").append(plant)
    out: bytes = etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
    return out, b.report


# --- validation -----------------------------------------------------------------------

@cache
def _schema() -> etree.XMLSchema:
    parser = etree.XMLParser(load_dtd=True, resolve_entities=True, no_network=True)
    return etree.XMLSchema(etree.parse(str(XSD), parser))


@cache
def _vocab() -> dict[str, Any]:
    data = json.loads(VOCAB.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data["classes"]  # type: ignore[no-any-return]


@cache
def _all_props(cls: str) -> frozenset[str]:
    v = _vocab()
    props = set(v.get(cls, {}).get("properties", []))
    for s in v.get(cls, {}).get("supertypes", []):
        props |= _all_props(s)
    return frozenset(props)


def validate_dexpi(xml: bytes, check_data_properties: bool = True) -> list[str]:
    """XSD validation plus DEXPI vocabulary checks. Returns error messages."""
    doc = etree.fromstring(xml)
    schema = _schema()
    errors = [] if schema.validate(doc) else [f"XSD: {e.message}" for e in schema.error_log]
    v = _vocab()
    ids = {o.get("id") for o in doc.iter("Object") if o.get("id")}
    for o in doc.iter("Object"):
        t = o.get("type", "")
        if t not in v:
            errors.append(f"unknown class {t}")
            continue
        if v[t]["abstract"]:
            errors.append(f"abstract class instantiated: {t}")
        allowed = _all_props(t)
        for child in o:
            if child.tag not in ("Components", "References", "Data"):
                continue
            if child.tag == "Data" and not check_data_properties:
                continue  # data properties also come from templates, not modelled here
            prop = child.get("property")
            if prop not in allowed:
                errors.append(f"{t} has no property {prop!r}")
            if child.tag == "References":
                for r in (child.get("objects") or "").split():
                    if r.startswith("#") and r[1:] not in ids:
                        errors.append(f"dangling reference {r} ({t}.{prop})")
    return errors
