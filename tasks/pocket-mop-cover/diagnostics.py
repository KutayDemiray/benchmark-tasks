"""Audit native per-material physics without modifying simulator state."""

from task_paths import OUTPUT_ROOT

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from blade_geometry import node_penetration, triangle_penetration
from mesh import tet_boundary
from self_contact import crossings, separating_translations


def edges_of(elements):
    """Return unique edges of triangle or tetrahedral elements."""
    candidates = [elements[:, [i, j]] for i, j in itertools.combinations(range(elements.shape[1]), 2)]
    return np.unique(np.sort(np.concatenate(candidates), axis=1), axis=0)


def audit_run(run):
    """Compute physical diagnostics from a completed, independently saved trace."""
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "completed", "Need a completed recording"
    request = json.loads((run / "request.json").read_text())
    profile = request.get("blade_profile", "square")
    config = json.loads((run / "source/parameters.json").read_text())
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    dt = 1 / request["physics_hz"]
    selected = np.asarray([row["repeat"] == rows[-1]["repeat"] and
                           row["time_s"] >= rows[-1]["time_s"] - 2 + dt / 2 for row in rows])
    hold_indices = np.flatnonzero(selected)
    complete_hold = len(hold_indices) == round(2 / dt)
    blade = np.asarray([row["measured_blade_pose_xyzw"][:3] for row in rows])
    native = np.load(run / "native-states.npz")
    parts = []
    index = 0
    while f"rest_{index}" in native:
        rest, elements = native[f"rest_{index}"], native[f"elements_{index}"]
        points, velocities = native[f"positions_{index}"], native[f"velocities_{index}"]
        assert len(points) == len(rows) == len(velocities)
        edges = edges_of(elements)
        lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
        finite = bool(np.isfinite(points).all() and np.isfinite(velocities).all())
        max_strain, min_ratio, inverted_steps = 0.0, None, 0
        for start in range(0, len(points), 64):
            part = points[start:start + 64]
            strain = np.linalg.norm(part[:, edges[:, 1]] - part[:, edges[:, 0]], axis=2) / lengths - 1
            max_strain = max(max_strain, float(np.abs(strain).max()))
            if elements.shape[1] == 4:
                rest_tets = rest[elements]
                determinants = np.linalg.det(rest_tets[:, 1:] - rest_tets[:, :1])
                tets = part[:, elements]
                ratio = np.linalg.det(tets[:, :, 1:] - tets[:, :, :1]) / determinants
                min_ratio = min(float(ratio.min()), min_ratio if min_ratio is not None else float("inf"))
                inverted_steps += int(np.any(ratio <= 0, axis=1).sum())
        speed = float(np.linalg.norm(velocities[selected], axis=2).max()) if complete_hold else None
        fd_speed = float(np.linalg.norm(np.diff(points[hold_indices], axis=0), axis=2).max() / dt) if complete_hold else None
        node_pen = node_penetration(points, blade[:,None], config["estimated_probe"]["blade_size_m"], profile)
        surface = elements if elements.shape[1] == 3 else tet_boundary(elements)
        triangle_steps = 0
        crossing_steps=0
        crossing_violation_steps=0
        crossing_span=0.
        crossing_depth=0.
        for step,(position, center) in enumerate(zip(points, blade, strict=True)):
            if elements.shape[1]==3:
                pairs,spans=crossings(position,surface)
                if len(pairs):
                    crossing_steps+=1
                    crossing_span=max(crossing_span,float(spans.max()))
                    depth=float(np.linalg.norm(separating_translations(position,surface,pairs),axis=1).max())
                    crossing_depth=max(crossing_depth,depth)
                    crossing_violation_steps+=int(depth>.001)
            if step%1000==0:
                progress=dict(part=index,step=step,total=len(points),self_crossing_violation_steps=crossing_violation_steps)
                (run/"audit-heartbeat.json").write_text(json.dumps(progress))
                print(json.dumps(progress),flush=True)
            triangle_steps += triangle_penetration(position, surface, center, config["estimated_probe"]["blade_size_m"], profile) > 0
        parts.append(dict(part=index, model="volume" if elements.shape[1] == 4 else "surface",
                          finite=finite, max_abs_edge_strain=max_strain,
                          min_tet_volume_ratio=min_ratio, inverted_steps=inverted_steps,
                          final_two_seconds_max_native_speed_m_s=speed,
                          final_two_seconds_max_difference_speed_m_s=fd_speed,
                          max_node_blade_penetration_m=node_pen,
                          max_ground_penetration_m=max(0.0, -float(points[:, :, 2].min())),
                          penetrating_triangle_steps=int(triangle_steps),self_crossing_steps=crossing_steps,
                          self_crossing_violation_steps=crossing_violation_steps,
                          self_crossing_normal_span_estimate_m=crossing_span,
                          self_crossing_penetration_upper_bound_m=crossing_depth))
        index += 1
    metrics_ok = complete_hold and all(
        p["finite"] and p["max_abs_edge_strain"] <= config["gate"]["max_abs_edge_strain"]
        and p["inverted_steps"] == 0 and p["penetrating_triangle_steps"] == 0 and p["self_crossing_violation_steps"]==0
        and p["max_ground_penetration_m"] <= config["gate"]["max_penetration_m"]
        and p["final_two_seconds_max_native_speed_m_s"] <= config["gate"]["settle_speed_m_s"]
        and p["final_two_seconds_max_difference_speed_m_s"] <= config["gate"]["settle_speed_m_s"]
        for p in parts)
    seam_gap = max(row["max_seam_gap_m"] for row in rows)
    physical_configuration = ((request.get("reinforced", False) or request.get("product_profile") == "base_micro")
                              and request.get("seam_filter", "element") != "pair"
                              and not request.get("detach_lip", False))
    report = dict(parts=parts, complete_final_two_second_window=complete_hold,
                  max_seam_gap_m=seam_gap, metrics_passed=bool(metrics_ok and seam_gap <= 0.001),
                  physical_configuration=physical_configuration,
                  diagnostic_stability_passed=bool(metrics_ok and seam_gap <= 0.001 and physical_configuration),
                  full_product_gate_passed=False,
                  scope="Native material/contact diagnostic only; grasping, insertion, and release need separate evidence.")
    (run / "physics-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT_ROOT)
    print(json.dumps(audit_run(args.run), indent=2))
