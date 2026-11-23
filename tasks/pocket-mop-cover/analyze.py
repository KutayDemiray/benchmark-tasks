"""Replay recorded states on the CPU and independently inspect pocket contact."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from measurements import closed_pocket_faces, pocket_seating, triangles_in_box
from mesh import edge_strain, surface_edges
from blade_geometry import node_penetration, triangle_penetration


def analyze(run):
    assert (run / "result.json").exists(), "Recording has not finished; wait for result.json"
    assert json.loads((run / "result.json").read_text())["status"] == "completed", "No completed native recording"
    data = np.load(run / "states.npz")
    trace = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    assert all("measured_blade_pose_xyzw" in row for row in trace), "Historical command-only recordings cannot establish actual tool contact"
    request = json.loads((run / "request.json").read_text())
    profile = request.get("blade_profile", "square")
    config = json.loads((run / "source/parameters.json").read_text())
    markers = json.loads((run / "mesh.json").read_text())["markers"]
    closed = closed_pocket_faces(data["rest"], data["faces"], markers)
    size = config["estimated_probe"]["blade_size_m"]
    cloth_edges = surface_edges(data["faces"])
    samples, max_strain, max_pen, max_motion_error, replay_error = [], 0.0, 0.0, 0.0, 0.0
    for index, (points, center, row) in enumerate(zip(data["positions"], data["blade_positions"], trace, strict=True)):
        strain = float(np.max(np.abs(edge_strain(data["rest"], points, data["faces"], edges=cloth_edges))))
        depth = node_penetration(points, center, size, profile)
        max_strain, max_pen = max(strain, max_strain), max(depth, max_pen)
        replay_error = max(replay_error, abs(strain - row["max_abs_edge_strain"]), abs(depth - row["max_blade_node_penetration_m"]))
        max_motion_error = max(max_motion_error, float(np.linalg.norm(center - row["rig_command_m"])))
        if index % max(1, request["physics_hz"] // 20) == 0 or index == len(trace) - 1:
            measurement = pocket_seating(points, closed, center, size, markers=markers, faces=data["faces"], profile=profile)
            measurement.update(step=index, time_s=row["time_s"], repeat=row["repeat"], phase=row["phase"],
                               penetrating_triangles=triangle_penetration(points, data["faces"], center, size, profile),
                               max_abs_edge_strain=strain)
            samples.append(measurement)
    digest = hashlib.sha256((run / "states.npz").read_bytes()).hexdigest()
    hold_indices = []
    for i, row in enumerate(trace):
        if row["repeat"] == trace[-1]["repeat"] and row["phase"] in ("tool_hold", "released_coupon") and row["time_s"] >= trace[-1]["time_s"] - 2:
            hold_indices.append(i)
    hold = [trace[i] for i in hold_indices]
    hold_speeds = [r["max_speed_m_s"] for r in hold]
    hold_speed = max(hold_speeds) if hold_speeds else None
    difference_speed = None
    if len(hold_indices) > 1:
        differences = np.diff(data["positions"][hold_indices], axis=0) * request["physics_hz"]
        difference_speed = float(np.linalg.norm(differences, axis=2).max())
    result = {"states_sha256": digest, "samples": samples, "replayed_physics_steps": len(trace),
              "metric_replay_max_abs_error": replay_error, "metric_replay_passed": replay_error < 1e-6,
              "max_abs_edge_strain": max_strain, "max_node_penetration_m": max_pen,
              "max_tool_tracking_error_m": max_motion_error,
              "triangle_penetration_sample_count": sum(s["penetrating_triangles"] > 0 for s in samples),
              "seat_sample_count": sum(s["seated_samples"] for s in samples),
              "final_two_seconds_max_node_speed_m_s": hold_speed,
              "final_two_seconds_max_finite_difference_speed_m_s": difference_speed,
              "hold_stationary": bool(hold_speeds and hold_speed < config["gate"]["settle_speed_m_s"]),
              "max_ground_penetration_m": max(0.0, -float(data["positions"][:, :, 2].min())),
              "final_seating": samples[-1], "benchmark_success": False,
              "independent_physics_replay": "not performed; this is independent measurement replay of recorded native states",
              "limitation": "Requires measured blade poses (r7 onward); historical command-only r5 files are rejected. Coupon checks do not validate full-task scoring."}
    result["analyzer_sources_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("analyze.py", "mesh.py", "measurements.py")}
    (run / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT_ROOT)
    analyze(args.run)
