"""SI-unit mechanism specification and passive, command-independent forces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
JOINTS = ("jaw", "fold_a", "fold_b")
BODIES = ("jaw_a", "jaw_b", "handle_a", "handle_b")


def load_spec(path=None):
    path = Path(path or ROOT / "mechanism.json")
    spec = json.loads(path.read_text())
    validate(spec)
    return spec


def validate(s):
    if s.get("schema_version") != 1 or set(s["joints"]) != set(JOINTS):
        raise ValueError("Unsupported mechanism schema or joint graph")
    if set(s["body_masses_kg"]) != set(BODIES):
        raise ValueError("Exactly four tool bodies are required")
    if any(not np.isfinite(m) or m <= 0 for m in s["body_masses_kg"].values()):
        raise ValueError("Invalid body mass")
    if abs(sum(s["body_masses_kg"].values()) - s["published_mass_kg"]) > 1e-6:
        raise ValueError("Link masses do not sum to the declared total")
    for name, j in s["joints"].items():
        lo, hi = j["limits_rad"]
        k = np.asarray(j["potential_knots"], float)
        if not lo < hi or not lo <= j["reset_rad"] <= hi:
            raise ValueError(f"Invalid limits/reset: {name}")
        if k.ndim != 2 or k.shape[1] != 2 or len(k) < 2 or not np.isfinite(k).all():
            raise ValueError(f"Invalid potential: {name}")
        if np.any(np.diff(k[:, 0]) <= 0) or abs(k[0, 0]-lo) > 1e-8 or abs(k[-1, 0]-hi) > 1e-8:
            raise ValueError(f"Potential must cover joint limits: {name}")
        if min(j["coulomb_nm"], j["viscous_nms"]) < 0 or j["velocity_regularization_rads"] <= 0:
            raise ValueError(f"Non-passive dissipation: {name}")
    if any(not np.isfinite(v) or v <= 0 for k, v in s["geometry"].items() if k not in ("fold_pivot_x", "handle_tail_x")):
        raise ValueError("Invalid geometry")


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def potential(q, joint):
    """C1 smoothstep interpolation of potential energy, and its derivative."""
    k = np.asarray(joint["potential_knots"], float)
    q = np.asarray(q, float)
    i = np.clip(np.searchsorted(k[:, 0], q, side="right")-1, 0, len(k)-2)
    width = k[i+1, 0]-k[i, 0]
    x = np.clip((q-k[i, 0])/width, 0, 1)
    delta = k[i+1, 1]-k[i, 1]
    u = k[i, 1]+delta*x*x*(3-2*x)
    du = delta*6*x*(1-x)/width
    return u, du


def effort(q, velocity, joint):
    _, du = potential(q, joint)
    velocity = np.asarray(velocity, float)
    loss = joint["coulomb_nm"]*np.tanh(velocity/joint["velocity_regularization_rads"])
    loss += joint["viscous_nms"]*velocity
    return -du-loss


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def transform(pos=(0, 0, 0), rot=None):
    t = np.eye(4)
    t[:3, :3] = np.eye(3) if rot is None else rot
    t[:3, 3] = pos
    return t


def link_transforms(spec, q):
    """Body poses relative to jaw_a. All joint anchors are parent-local."""
    g = spec["geometry"]
    a = (g["fold_pivot_x"], -g["fold_pivot_half_span"], 0)
    b = (g["fold_pivot_x"], g["fold_pivot_half_span"], 0)
    jb = transform(rot=rz(q["jaw"]))
    return {"jaw_a": np.eye(4), "jaw_b": jb,
            "handle_a": transform(a, rz(q["fold_a"])),
            "handle_b": jb @ transform(b, rz(q["fold_b"]))}


def reset_joints(spec):
    return {k: spec["joints"][k]["reset_rad"] for k in JOINTS}


def jaw_gap(spec, qjaw):
    g = spec["geometry"]
    a = np.array([g["jaw_tip_x"]-g["bite_from_tip"], g["jaw_tip_half_gap"], 0.])
    b = rz(qjaw) @ (a*np.array([1., -1., 1.]))
    return float(np.linalg.norm((a-b)[:2]))


def jaw_angle_for_gap(spec, gap):
    lo, hi = spec["joints"]["jaw"]["limits_rad"]
    if not jaw_gap(spec, hi) <= gap <= jaw_gap(spec, lo):
        raise ValueError("Requested gap exceeds measured jaw range")
    for _ in range(60):
        mid = (lo+hi)/2
        if jaw_gap(spec, mid) > gap:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2
