"""Reviewer corrections applied to a predicted graph.

Operations are validated pydantic models; `apply_ops` is pure (returns a new Drawing)
and raises ValueError on any invalid operation, so a PATCH is all-or-nothing.
Human-touched elements get confidence 1.0 (reviewed).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from pid.io.drawing import Drawing, Edge, LineType

NODE_CLASSES = ("general", "tank", "valve", "instrumentation", "pump", "arrow",
                "inlet_outlet", "crossing", "ankle")


class RelabelNode(BaseModel):
    op: Literal["relabel_node"]
    id: str
    cls: Literal["general", "tank", "valve", "instrumentation", "pump", "arrow",
                 "inlet_outlet", "crossing", "ankle"]


class DeleteNode(BaseModel):
    op: Literal["delete_node"]
    id: str


class DeleteEdge(BaseModel):
    op: Literal["delete_edge"]
    id: str


class AddEdge(BaseModel):
    op: Literal["add_edge"]
    source: str
    target: str
    line_type: LineType = LineType.SOLID


class SetLineType(BaseModel):
    op: Literal["set_line_type"]
    id: str
    line_type: LineType


class Confirm(BaseModel):
    op: Literal["confirm"]
    id: str  # node or edge id


Op = Annotated[RelabelNode | DeleteNode | DeleteEdge | AddEdge | SetLineType | Confirm,
               Field(discriminator="op")]


class CorrectionRequest(BaseModel):
    ops: list[Op] = Field(min_length=1, max_length=1000)


def apply_ops(d: Drawing, ops: Sequence[BaseModel]) -> Drawing:
    out = d.model_copy(deep=True)
    for k, op in enumerate(ops):
        nodes = {s.id: s for s in out.symbols}
        edges = {e.id: e for e in out.edges}
        where = f"op {k} ({type(op).__name__})"
        if isinstance(op, RelabelNode):
            if op.id not in nodes:
                raise ValueError(f"{where}: unknown node {op.id!r}")
            nodes[op.id].cls = op.cls
            nodes[op.id].score = 1.0
        elif isinstance(op, DeleteNode):
            if op.id not in nodes:
                raise ValueError(f"{where}: unknown node {op.id!r}")
            out.symbols = [s for s in out.symbols if s.id != op.id]
            out.edges = [e for e in out.edges if op.id not in (e.source, e.target)]
        elif isinstance(op, DeleteEdge):
            if op.id not in edges:
                raise ValueError(f"{where}: unknown edge {op.id!r}")
            out.edges = [e for e in out.edges if e.id != op.id]
        elif isinstance(op, AddEdge):
            for nid in (op.source, op.target):
                if nid not in nodes:
                    raise ValueError(f"{where}: unknown node {nid!r}")
            if op.source == op.target:
                raise ValueError(f"{where}: self-loop")
            if any(e.key() == frozenset((op.source, op.target)) for e in out.edges):
                raise ValueError(f"{where}: edge already exists")
            num = len(out.edges)
            while f"u{num}" in edges:
                num += 1
            out.edges.append(Edge(id=f"u{num}", source=op.source, target=op.target,
                                  line_type=op.line_type, score=1.0))
        elif isinstance(op, SetLineType):
            if op.id not in edges:
                raise ValueError(f"{where}: unknown edge {op.id!r}")
            edges[op.id].line_type = op.line_type
            edges[op.id].score = 1.0
        elif isinstance(op, Confirm):
            if op.id in nodes:
                nodes[op.id].score = 1.0
            elif op.id in edges:
                edges[op.id].score = 1.0
            else:
                raise ValueError(f"{where}: unknown node or edge {op.id!r}")
        else:  # pragma: no cover - the discriminated union makes this unreachable
            raise ValueError(f"{where}: unsupported operation")
    return Drawing.model_validate(out.model_dump())
