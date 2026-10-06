"""Calibrated per-edge / per-node confidence (Milestone 3).

Each edge produced by `graph.assemble` carries geometric features (FEATURES). A
logistic regression over the standardized features, fitted on validation drawings
with label "edge is a true positive", gives a calibrated probability:

    p(edge correct) = sigmoid(b + sum_i w_i * (x_i - mu_i) / sd_i)

Structural nodes get the mean confidence of their incident edges; symbols keep the
detector score. Implemented in numpy (Newton/IRLS with L2) to avoid a scikit-learn
dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pid.io.drawing import Drawing

F = NDArray[np.float64]
FEATURES: tuple[str, ...] = ("is_port", "nonsolid", "fill", "width", "log_len", "slack",
                             "n_ankle_ends", "snap_dist", "sym_score", "deg_min", "deg_max")
STRUCTURAL = ("crossing", "ankle")


def vectorize(rows: Sequence[Mapping[str, float]]) -> F:
    return np.array([[float(r.get(k, 0.0)) for k in FEATURES] for r in rows], dtype=np.float64
                    ).reshape(-1, len(FEATURES))


def _sigmoid(z: F) -> F:
    out: F = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return out


@dataclass
class ConfidenceModel:
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def predict(self, X: F) -> F:
        Z = (X - np.asarray(self.mean)) / np.asarray(self.std)
        return _sigmoid(Z @ np.asarray(self.coef) + self.intercept)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> ConfidenceModel:
        m = cls(**json.loads(path.read_text()))
        if tuple(m.features) != FEATURES:
            raise ValueError(f"{path}: feature list does not match FEATURES")
        return m


def fit(X: F, y: F, l2: float = 1.0, iters: int = 50) -> ConfidenceModel:
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[0] = 0.0  # no penalty on the intercept
    for _ in range(iters):
        p = _sigmoid(Z @ w)
        g = Z.T @ (p - y) + reg * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return ConfidenceModel(mean=mu.tolist(), std=sd.tolist(), coef=w[1:].tolist(),
                           intercept=float(w[0]))


def expected_calibration_error(p: F, y: F, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """ECE with equal-width bins; also returns the reliability table."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        conf, acc, n = float(p[sel].mean()), float(y[sel].mean()), int(sel.sum())
        ece += n / len(p) * abs(conf - acc)
        table.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                      "confidence": conf, "accuracy": acc, "n": n})
    return float(ece), table


def apply_confidence(d: Drawing, features: Mapping[str, Mapping[str, float]],
                     model: ConfidenceModel) -> Drawing:
    """Set edge scores from the model and structural node scores from their edges."""
    if not d.edges:
        return d
    p = model.predict(vectorize([features[e.id] for e in d.edges]))
    incident: dict[str, list[float]] = {}
    for e, pe in zip(d.edges, p, strict=True):
        e.score = float(pe)
        incident.setdefault(e.source, []).append(float(pe))
        incident.setdefault(e.target, []).append(float(pe))
    for s in d.symbols:
        if s.cls in STRUCTURAL and s.id in incident:
            s.score = float(np.mean(incident[s.id]))
    return d
