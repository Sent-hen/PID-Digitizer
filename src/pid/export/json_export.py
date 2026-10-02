"""Minimal JSON export (Milestone 1): nodes and edges only.

The versioned schema with metadata, ports, texts and confidences is Milestone 4.
"""

from __future__ import annotations

from typing import Any

from pid.io.drawing import Drawing

SCHEMA_VERSION = "0.1"


def export_minimal(d: Drawing) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "drawing": {"id": d.id, "subset": d.subset, "width": d.width, "height": d.height},
        "nodes": [{"id": s.id, "class": s.cls, "bbox": [round(v, 2) for v in s.bbox.as_xyxy()],
                   "score": round(s.score, 4)} for s in d.symbols],
        "edges": [{"id": e.id, "source": e.source, "target": e.target,
                   "line_type": e.line_type.value} for e in d.edges],
    }
