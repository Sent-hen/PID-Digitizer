"""Canonical in-memory representation of an annotated (or predicted) P&ID drawing.

Every loader converts its source format into a `Drawing`; every downstream stage
(evaluation, export, overlays) consumes only this model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class LineType(StrEnum):
    SOLID = "solid"
    NON_SOLID = "non-solid"


class BBox(BaseModel):
    """Axis-aligned box in full-drawing pixel coordinates (x right, y down)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        # Some source annotations have min/max swapped; normalize instead of failing.
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin
        return self

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class Port(BaseModel):
    """A connection point on a symbol. PID2Graph has no port annotations; reserved."""

    id: str
    x: float
    y: float


class Symbol(BaseModel):
    """A graph node: a P&ID symbol, or a structural node (crossing / ankle)."""

    id: str
    cls: str
    bbox: BBox
    score: float = 1.0
    ports: list[Port] = Field(default_factory=list)


class Text(BaseModel):
    """Recognized/annotated text. PID2Graph has no text ground truth; reserved."""

    id: str
    text: str
    bbox: BBox
    rotation: int = 0
    score: float = 1.0


class Edge(BaseModel):
    """An undirected connection between two symbols."""

    id: str
    source: str
    target: str
    line_type: LineType = LineType.SOLID
    score: float = 1.0

    def key(self) -> frozenset[str]:
        return frozenset((self.source, self.target))


class Drawing(BaseModel):
    id: str
    subset: str = ""
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    symbols: list[Symbol] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    texts: list[Text] = Field(default_factory=list)
    ignore_regions: list[BBox] = Field(
        default_factory=list,
        description="Regions (e.g. border/title block) that carry no graph content.",
    )

    @model_validator(mode="after")
    def _edges_reference_symbols(self) -> Drawing:
        ids = {s.id for s in self.symbols}
        if len(ids) != len(self.symbols):
            raise ValueError(f"{self.id}: duplicate symbol ids")
        for e in self.edges:
            if e.source not in ids or e.target not in ids:
                raise ValueError(f"{self.id}: edge {e.id} references unknown symbol")
        return self

    def symbol_by_id(self) -> dict[str, Symbol]:
        return {s.id: s for s in self.symbols}
