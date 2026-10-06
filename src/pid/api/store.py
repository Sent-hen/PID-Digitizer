"""SQLite-backed job state (single process, thread-safe)."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done | failed
    stage TEXT NOT NULL,
    progress REAL NOT NULL,
    error TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    n_nodes INTEGER,
    n_edges INTEGER,
    n_corrections INTEGER NOT NULL DEFAULT 0
)
"""


@dataclass
class Job:
    id: str
    filename: str
    status: str
    stage: str
    progress: float
    error: str | None
    created: float
    updated: float
    width: int | None
    height: int | None
    n_nodes: int | None
    n_edges: int | None
    n_corrections: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(SCHEMA)
            self._db.commit()

    def create(self, job_id: str, filename: str, width: int, height: int) -> Job:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, filename, status, stage, progress, created, updated, "
                "width, height) VALUES (?, ?, 'queued', 'queued', 0, ?, ?, ?, ?)",
                (job_id, filename, now, now, width, height))
            self._db.commit()
        job = self.get(job_id)
        assert job is not None
        return job

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            self._db.commit()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [Job(**dict(r)) for r in rows]

    def mark_interrupted(self) -> int:
        """Jobs left queued/running by a previous process can never finish."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status = 'failed', error = 'interrupted by server restart', "
                "updated = ? WHERE status IN ('queued', 'running')", (time.time(),))
            self._db.commit()
            return cur.rowcount
