"""FastAPI service: upload a P&ID image, get its graph as GraphML or DEXPI, review it.

Run:  uvicorn pid.api.app:app --host 0.0.0.0 --port 8000   (or `make serve`)

Endpoints (OpenAPI docs at /docs):
  POST  /jobs                      upload an image (multipart field `file`) -> job
  GET   /jobs                      recent jobs
  GET   /jobs/{id}                 status / progress
  GET   /jobs/{id}/graph           current graph as JSON (UI transport)
  PATCH /jobs/{id}/graph           apply reviewer corrections (append-only log)
  GET   /jobs/{id}/result          ?format=graphml|dexpi[&version=original|current]
  GET   /jobs/{id}/overlay         PNG overlay coloured by confidence
  GET   /jobs/{id}/image           downscaled input image for the review UI
  GET   /health                    liveness + model status
The review UI is served at /.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import ValidationError

from pid.api.corrections import CorrectionRequest, apply_ops
from pid.api.store import Job, JobStore
from pid.config import load_config, resolve
from pid.export.dexpi import to_dexpi
from pid.export.graphml import read_graphml, write_graphml
from pid.io.drawing import Drawing
from pid.io.overlay import render_overlay

DigitizeFn = Callable[[Path, str], Drawing]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"time": round(record.created, 3), "level": record.levelname,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


log = logging.getLogger("pid.api")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(JsonFormatter())
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def _event(msg: str, **fields: Any) -> None:
    log.info(msg, extra={"fields": fields})


@dataclass
class Settings:
    data_dir: Path
    weights: Path
    weights_sha1: str | None
    device: str
    max_upload_mb: float
    max_pixels: int
    allowed_formats: list[str]
    workers: int
    job_timeout_s: float
    overlay_max_side: int
    ui_image_max_side: int
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Settings:
        c = load_config("serve")
        return cls(
            data_dir=resolve(os.environ.get("PID_DATA_DIR", c["data_dir"])),
            weights=resolve(os.environ.get("PID_WEIGHTS", c["weights"])),
            weights_sha1=c.get("weights_sha1"),
            device=os.environ.get("PID_DEVICE", c.get("device", "auto")),
            max_upload_mb=float(c["max_upload_mb"]), max_pixels=int(c["max_pixels"]),
            allowed_formats=[f.upper() for f in c["allowed_formats"]],
            workers=int(c["workers"]), job_timeout_s=float(c["job_timeout_s"]),
            overlay_max_side=int(c["overlay_max_side"]),
            ui_image_max_side=int(c["ui_image_max_side"]))


def real_pipeline(settings: Settings) -> DigitizeFn:
    """Load the detector once and return the production digitize function."""
    import torch

    from pid.detect.model import TiledDetector
    from pid.pipeline import digitize

    if not settings.weights.exists():
        raise RuntimeError(f"detector weights not found at {settings.weights} "
                           "(see scripts/fetch_weights.py)")
    if settings.weights_sha1:
        digest = hashlib.sha1(settings.weights.read_bytes()).hexdigest()[:12]
        if digest != settings.weights_sha1:
            raise RuntimeError(f"weights sha1 {digest} != expected {settings.weights_sha1}")
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = TiledDetector(load_config("detect"), settings.weights, device=device)
    pcfg = load_config("pipeline")
    _event("model loaded", weights=str(settings.weights), device=device)

    def run(path: Path, job_id: str) -> Drawing:
        return digitize(path, pcfg, drawing_id=job_id, subset="upload", detector=detector)

    return run


class Service:
    def __init__(self, settings: Settings, digitize_fn: DigitizeFn | None) -> None:
        self.settings = settings
        self.digitize_fn = digitize_fn
        self.store = JobStore(settings.data_dir / "jobs.sqlite")
        self.pool = ThreadPoolExecutor(max_workers=settings.workers,
                                       thread_name_prefix="pid-job")
        self.model_error: str | None = None

    def job_dir(self, job_id: str) -> Path:
        return self.settings.data_dir / "jobs" / job_id

    def input_path(self, job_id: str) -> Path:
        hits = sorted(self.job_dir(job_id).glob("input.*"))
        if not hits:
            raise HTTPException(404, "input image missing")
        return hits[0]

    def current_graph_path(self, job_id: str) -> Path:
        d = self.job_dir(job_id)
        return d / "corrected.graphml" if (d / "corrected.graphml").exists() \
            else d / "result.graphml"

    def run_job(self, job_id: str) -> None:
        t0 = time.time()
        self.store.update(job_id, status="running", stage="digitizing", progress=0.1)
        _event("job started", job=job_id)
        try:
            if self.digitize_fn is None:
                raise RuntimeError(self.model_error or "model not loaded")
            drawing = self.digitize_fn(self.input_path(job_id), job_id)
            elapsed = time.time() - t0
            if elapsed > self.settings.job_timeout_s:
                raise TimeoutError(f"job exceeded {self.settings.job_timeout_s:.0f} s")
            self.store.update(job_id, stage="exporting", progress=0.9)
            write_graphml(drawing, self.job_dir(job_id) / "result.graphml")
            self.store.update(job_id, status="done", stage="done", progress=1.0,
                              n_nodes=len(drawing.symbols), n_edges=len(drawing.edges))
            _event("job done", job=job_id, seconds=round(elapsed, 2),
                   nodes=len(drawing.symbols), edges=len(drawing.edges))
        except Exception as exc:  # noqa: BLE001 - every failure is recorded on the job
            self.store.update(job_id, status="failed", stage="failed", error=repr(exc))
            _event("job failed", job=job_id, error=repr(exc))


def create_app(settings: Settings | None = None,
               digitize_fn: DigitizeFn | None = None,
               load_model: bool = True) -> FastAPI:
    settings = settings or Settings.load()
    holder: dict[str, Service] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        fn = digitize_fn
        svc = Service(settings, fn)
        n = svc.store.mark_interrupted()
        if n:
            _event("marked interrupted jobs failed", count=n)
        if fn is None and load_model:
            try:
                svc.digitize_fn = real_pipeline(settings)
            except Exception as exc:  # noqa: BLE001 - surfaced via /health
                svc.model_error = repr(exc)
                _event("model load failed", error=repr(exc))
        holder["svc"] = svc
        yield
        svc.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="P&ID Digitizer", version="1.0.0", lifespan=lifespan,
                  description="Raster P&ID -> graph (GraphML / DEXPI-structured XML).")

    def svc() -> Service:
        return holder["svc"]

    def job_or_404(job_id: str) -> Job:
        job = svc().store.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job

    def done_or_409(job_id: str) -> Job:
        job = job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        return job

    @app.get("/health")
    def health() -> dict[str, Any]:
        s = svc()
        return {"status": "ok", "model_loaded": s.digitize_fn is not None,
                "model_error": s.model_error}

    @app.post("/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        s = svc()
        limit = int(s.settings.max_upload_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"upload exceeds {s.settings.max_upload_mb:g} MB")
        try:
            Image.MAX_IMAGE_PIXELS = 2 * s.settings.max_pixels  # decompression-bomb guard
            with Image.open(io.BytesIO(data)) as im:
                fmt, (w, h) = (im.format or "").upper(), im.size
                im.verify()
        except Exception as exc:
            raise HTTPException(415, f"not a readable image: {exc}") from exc
        if fmt not in s.settings.allowed_formats:
            raise HTTPException(415, f"format {fmt} not allowed ({s.settings.allowed_formats})")
        if w * h > s.settings.max_pixels:
            raise HTTPException(413, f"image has {w * h} pixels (max {s.settings.max_pixels})")
        job_id = uuid.uuid4().hex[:12]
        d = s.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        ext = {"JPEG": "jpg", "TIFF": "tif"}.get(fmt, fmt.lower())
        (d / f"input.{ext}").write_bytes(data)
        safe_name = Path(file.filename or "upload").name[:200]
        job = s.store.create(job_id, safe_name, w, h)
        s.pool.submit(s.run_job, job_id)
        _event("job queued", job=job_id, filename=safe_name, width=w, height=h,
               bytes=len(data))
        return job.as_dict()

    @app.get("/jobs")
    def list_jobs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        return [j.as_dict() for j in svc().store.list(limit)]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return job_or_404(job_id).as_dict()

    @app.get("/jobs/{job_id}/graph")
    def get_graph(job_id: str) -> JSONResponse:
        done_or_409(job_id)
        d = read_graphml(svc().current_graph_path(job_id))
        return JSONResponse(json.loads(d.model_dump_json()))

    @app.patch("/jobs/{job_id}/graph")
    def patch_graph(job_id: str, body: dict[str, Any]) -> JSONResponse:
        done_or_409(job_id)
        try:
            req = CorrectionRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(422, json.loads(exc.json())) from exc
        s = svc()
        path = s.current_graph_path(job_id)
        current = read_graphml(path)
        try:
            corrected = apply_ops(current, req.ops)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        d = s.job_dir(job_id)
        with (d / "corrections.jsonl").open("a", encoding="utf-8") as f:  # append-only log
            f.write(json.dumps({"time": time.time(),
                                "ops": [op.model_dump(mode="json") for op in req.ops]}) + "\n")
        write_graphml(corrected, d / "corrected.graphml")
        job = job_or_404(job_id)
        s.store.update(job_id, n_corrections=job.n_corrections + len(req.ops),
                       n_nodes=len(corrected.symbols), n_edges=len(corrected.edges))
        _event("graph corrected", job=job_id, ops=len(req.ops))
        return JSONResponse(json.loads(corrected.model_dump_json()))

    @app.get("/jobs/{job_id}/result")
    def get_result(job_id: str, format: str = Query("graphml", pattern="^(graphml|dexpi)$"),
                   version: str = Query("current", pattern="^(current|original)$")
                   ) -> Response:
        done_or_409(job_id)
        s = svc()
        path = (s.job_dir(job_id) / "result.graphml" if version == "original"
                else s.current_graph_path(job_id))
        if format == "graphml":
            return FileResponse(path, media_type="application/graphml+xml",
                                filename=f"{job_id}.graphml")
        xml, _ = to_dexpi(read_graphml(path))
        return Response(xml, media_type="application/xml", headers={
            "Content-Disposition": f'attachment; filename="{job_id}.dexpi.xml"'})

    @app.get("/jobs/{job_id}/overlay")
    def get_overlay(job_id: str, color_by: str = Query("confidence",
                                                      pattern="^(confidence|class)$")
                    ) -> Response:
        done_or_409(job_id)
        s = svc()
        d = read_graphml(s.current_graph_path(job_id))
        with Image.open(s.input_path(job_id)) as im:
            img = render_overlay(d, image=im, max_side=s.settings.overlay_max_side,
                                 color_by=color_by)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/jobs/{job_id}/image")
    def get_image(job_id: str) -> Response:
        job_or_404(job_id)
        s = svc()
        cache = s.job_dir(job_id) / "ui.jpg"
        with Image.open(s.input_path(job_id)) as im:
            w, h = im.size
            scale = min(1.0, s.settings.ui_image_max_side / max(w, h))
            if not cache.exists():
                small = im.convert("L").resize((round(w * scale), round(h * scale)))
                small.save(cache, quality=85)
        return FileResponse(cache, media_type="image/jpeg",
                            headers={"X-Scale": f"{scale:.6f}"})

    if UI_DIR.exists():
        app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    return app


app = create_app()
