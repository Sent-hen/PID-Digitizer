"""API tests (pytest + httpx via FastAPI's TestClient). The pipeline is replaced by a
fake digitize function, so these run on CPU without weights, torch or the dataset."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from lxml import etree
from PIL import Image

from pid.api.app import Settings, create_app
from pid.export.dexpi import validate_dexpi
from pid.export.graphml import validate_graphml
from pid.io.drawing import BBox, Drawing, Edge, Symbol


def fake_digitize(path: Path, job_id: str) -> Drawing:
    def s(i: str, cls: str, x: float, y: float, w: float = 8, score: float = 0.9) -> Symbol:
        return Symbol(id=i, cls=cls, score=score,
                      bbox=BBox(xmin=x, ymin=y, xmax=x + w, ymax=y + w))

    syms = [s("s0", "tank", 10, 10, 40), s("s1", "valve", 120, 20, 20),
            s("a0", "ankle", 50, 26), s("a1", "ankle", 116, 26), s("a2", "ankle", 140, 26),
            s("c0", "crossing", 200, 26, score=0.4), s("s2", "general", 190, 100, 30)]
    pairs = [("s0", "a0"), ("a0", "a1"), ("a1", "s1"), ("s1", "a2"), ("a2", "c0")]
    edges = [Edge(id=f"e{k}", source=u, target=v, score=0.95 if k < 4 else 0.3)
             for k, (u, v) in enumerate(pairs)]
    with Image.open(path) as im:
        w, h = im.size
    return Drawing(id=job_id, subset="upload", width=w, height=h, image_path=str(path),
                   symbols=syms, edges=edges)


def png(w: int = 300, h: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> Any:
    settings = Settings(data_dir=tmp_path, weights=tmp_path / "none.pt", weights_sha1=None,
                        device="cpu", max_upload_mb=1, max_pixels=1_000_000,
                        allowed_formats=["PNG", "JPEG"], workers=1, job_timeout_s=60,
                        overlay_max_side=400, ui_image_max_side=200)
    app = create_app(settings, digitize_fn=fake_digitize)
    with TestClient(app) as c:
        yield c


def wait_done(c: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        j: dict[str, Any] = c.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def upload(c: TestClient, data: bytes, name: str = "sheet.png") -> Any:
    return c.post("/jobs", files={"file": (name, data, "image/png")})


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_upload_poll_and_results(client: TestClient) -> None:
    r = upload(client, png())
    assert r.status_code == 202
    job = r.json()
    assert job["status"] == "queued" and job["width"] == 300
    done = wait_done(client, job["id"])
    assert done["status"] == "done" and done["n_edges"] == 5

    g = client.get(f"/jobs/{job['id']}/result", params={"format": "graphml"})
    assert g.status_code == 200 and validate_graphml(g.content) == []
    x = client.get(f"/jobs/{job['id']}/result", params={"format": "dexpi"})
    assert x.status_code == 200 and validate_dexpi(x.content) == []
    types = [o.get("type") for o in etree.fromstring(x.content).iter("Object")]
    assert "Plant/ProcessEquipment.Tank" in types and "Plant/Piping.OperatedValve" in types

    ov = client.get(f"/jobs/{job['id']}/overlay")
    assert ov.status_code == 200 and ov.headers["content-type"] == "image/png"
    im = client.get(f"/jobs/{job['id']}/image")
    assert im.status_code == 200 and float(im.headers["x-scale"]) == pytest.approx(200 / 300)
    assert any(j["id"] == job["id"] for j in client.get("/jobs").json())


def test_correction_round_trip(client: TestClient) -> None:
    job_id = upload(client, png()).json()["id"]
    wait_done(client, job_id)
    ops = {"ops": [
        {"op": "delete_edge", "id": "e4"},                                  # spurious link
        {"op": "relabel_node", "id": "s2", "cls": "pump"},                  # class fix
        {"op": "add_edge", "source": "a2", "target": "s2", "line_type": "non-solid"},
        {"op": "confirm", "id": "c0"},
    ]}
    r = client.patch(f"/jobs/{job_id}/graph", json=ops)
    assert r.status_code == 200, r.text
    g = r.json()
    assert {e["id"] for e in g["edges"]} == {"e0", "e1", "e2", "e3", "u4"}
    added = next(e for e in g["edges"] if e["id"] == "u4")
    assert added["line_type"] == "non-solid" and added["score"] == 1.0
    assert next(s for s in g["symbols"] if s["id"] == "s2")["cls"] == "pump"
    # current result reflects the corrections; the original is preserved
    cur = client.get(f"/jobs/{job_id}/graph").json()
    assert cur == g
    orig = client.get(f"/jobs/{job_id}/result", params={"version": "original"}).content
    assert b'id="e4"' in orig
    assert client.get(f"/jobs/{job_id}").json()["n_corrections"] == 4
    # the DEXPI export of the corrected graph is still valid and now has a pump
    x = client.get(f"/jobs/{job_id}/result", params={"format": "dexpi"}).content
    assert validate_dexpi(x) == [] and b"ProcessEquipment.Pump" in x


def test_invalid_correction_is_atomic(client: TestClient) -> None:
    job_id = upload(client, png()).json()["id"]
    wait_done(client, job_id)
    r = client.patch(f"/jobs/{job_id}/graph", json={"ops": [
        {"op": "delete_edge", "id": "e0"}, {"op": "delete_edge", "id": "nope"}]})
    assert r.status_code == 400
    assert len(client.get(f"/jobs/{job_id}/graph").json()["edges"]) == 5  # nothing applied
    bad = client.patch(f"/jobs/{job_id}/graph", json={"ops": [{"op": "explode", "id": "x"}]})
    assert bad.status_code == 422


def test_upload_validation(client: TestClient) -> None:
    assert upload(client, b"not an image", "x.png").status_code == 415
    gif = io.BytesIO()
    Image.new("L", (10, 10)).save(gif, format="GIF")
    assert upload(client, gif.getvalue(), "x.gif").status_code == 415      # format not allowed
    assert upload(client, png(1200, 1000)).status_code == 413              # too many pixels
    assert upload(client, b"0" * (1024 * 1024 + 10)).status_code == 413   # too large


def test_unknown_and_unfinished_jobs(client: TestClient) -> None:
    assert client.get("/jobs/doesnotexist").status_code == 404
    assert client.get("/jobs/doesnotexist/graph").status_code == 404


def test_failed_job_reports_error(tmp_path: Path) -> None:
    def boom(path: Path, job_id: str) -> Drawing:
        raise RuntimeError("pipeline crashed")

    settings = Settings(data_dir=tmp_path, weights=tmp_path / "x", weights_sha1=None,
                        device="cpu", max_upload_mb=1, max_pixels=1_000_000,
                        allowed_formats=["PNG"], workers=1, job_timeout_s=60,
                        overlay_max_side=400, ui_image_max_side=200)
    with TestClient(create_app(settings, digitize_fn=boom)) as c:
        job_id = upload(c, png()).json()["id"]
        j = wait_done(c, job_id)
        assert j["status"] == "failed" and "pipeline crashed" in j["error"]
        assert c.get(f"/jobs/{job_id}/result").status_code == 409


def test_ui_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "<canvas" in r.text
