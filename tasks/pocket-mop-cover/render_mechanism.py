"""Render recorded rigid-body geometry and poses without running new physics."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from audit_mechanism import capsule_proxy


def load_run(run):
    """Load immutable native states and the matching collider geometry."""
    request = json.loads((run / "request.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "completed"
    metadata = json.loads((run / "mechanism.json").read_text())
    data = np.load(run / "states.npz")
    names = list(data["body_names"])
    bodies = data["body_poses"]
    shapes = list(metadata["colliders"])
    if "pedal_pusher" in metadata:
        for name, size in [("Finger", [.020, .025, .010]), ("Base", [.020, .020, .030])]:
            index = list(data["pusher_body_names"]).index(name)
            bodies = np.concatenate((bodies, data["pusher_poses"][:, index:index+1]), axis=1)
            names.append("Pusher" + name)
            points = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * size
            shapes.append(dict(body="Pusher" + name, points=points, enabled=True))
    joints = data["joint_positions"]
    joint_names = list(data["joint_names"])
    angles = np.rad2deg(abs(joints[:, [joint_names.index("HingeLeft"), joint_names.index("HingeRight")]])).max(1)
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    return dict(run=run, request=request, names=names, bodies=bodies, shapes=shapes,
                angles=angles, phases=[row["phase"] for row in rows])


def render(positive, negative, output):
    """Save a labelled, actual-speed side-view replay and its provenance."""
    runs = [load_run(positive), load_run(negative)]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f4f6f8")
    grid = fig.add_gridspec(2, 2, height_ratios=[2.1, 1], hspace=.38, wspace=.14)
    colors = dict(Bridge="#607080", Left="#1677b5", Right="#1677b5",
                  Rocker="#db493c", Clip="#e2ab24", Handle="#8a929b",
                  Connector="#8a929b", PusherFinger="#9939ac", PusherBase="#9939ac")
    fig.suptitle("Passive frame latch: lock, retain, then release", fontsize=19, y=.97)
    fig.text(.5, .91, "Recorded native poses and collision geometry • estimated mechanism • laboratory fixtures, no robot or cloth",
             ha="center", fontsize=10)
    lower = np.array([np.inf, np.inf]); upper = -lower
    for run in runs:
        for shape in run['shapes']:
            vertices = capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
            index = run['names'].index(shape['body'])
            poses = run['bodies'][::run['request']['physics_hz']//20, index]
            rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
            world = np.einsum('tij,vj->tvi', rotations, vertices) + poses[:, None, :3]
            projected = world[:, :, [0, 2]]
            lower = np.minimum(lower, projected.min(axis=(0, 1)))
            upper = np.maximum(upper, projected.max(axis=(0, 1)))
    lower[1] = min(lower[1], 0.)
    lower -= .015; upper += .015
    panels = []
    for column, (run, title) in enumerate(zip(runs, ["Latch contact enabled", "Control: latch contact disabled"])):
        ax = fig.add_subplot(grid[0, column])
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal", xlabel="Frame length (m)", ylabel="Height (m)")
        ax.set_title(title, fontsize=13)
        ax.axhline(0, color="#65717c", lw=3)
        ax.grid(alpha=.15)
        caption = ax.text(.02, .98, "", transform=ax.transAxes, va="top", fontsize=9,
                          bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
        patches = []
        for shape in run["shapes"]:
            points = capsule_proxy(shape["capsule"])[0] if "capsule" in shape else np.asarray(shape["points"])
            enabled = shape["enabled"]
            patch = Polygon(np.zeros((3, 2)), closed=True,
                            facecolor=colors.get(shape["body"], "#667788") if enabled else "none",
                            edgecolor="#48515b" if enabled else "#df8b2e", linewidth=.5,
                            linestyle="-" if enabled else ":")
            ax.add_patch(patch)
            patches.append((patch, points, run["names"].index(shape["body"])))
        panels.append((caption, patches))
    plot = fig.add_subplot(grid[1, :])
    plot.set(xlim=(0, 36), ylim=(-2, 60), xlabel="Recorded time (s)", ylabel="Wing angle (degrees)")
    plot.axhline(3, color="#268357", ls="--", lw=1, label="Retention limit: 3°")
    plot.axvspan(10, 24, color="#268357", alpha=.08, label="Audited retention window")
    for run, color, label in zip(runs, ["#1677b5", "#df8b2e"], ["Latch enabled", "Control"]):
        time = (np.arange(len(run["angles"])) + 1) / run["request"]["physics_hz"]
        plot.plot(time[::24], run["angles"][::24], color=color, lw=1.4, label=label)
    cursor = plot.axvline(0, color="#27323b", lw=1)
    plot.legend(loc="upper center", ncol=4, fontsize=9)
    plot.grid(alpha=.15)
    timestamp = fig.text(.5, .025, "", ha="center", fontsize=11)
    fig.subplots_adjust(top=.84, bottom=.12, left=.075, right=.97)
    fps = 20
    duration = len(runs[0]["bodies"]) / runs[0]["request"]["physics_hz"]
    frames = round(duration * fps)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(frames):
            t = (frame + 1) / fps
            for run, (caption, patches) in zip(runs, panels):
                index = min(round(t * run["request"]["physics_hz"]) - 1, len(run["bodies"]) - 1)
                for patch, vertices, body in patches:
                    pose = run["bodies"][index, body]
                    world = vertices @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
                    projected = world[:, [0, 2]]
                    hull = ConvexHull(projected)
                    patch.set_xy(projected[hull.vertices])
                    patch.set_zorder(2 - float(world[:, 1].mean()))
                phase = run["phases"][index].replace("_", " ")
                ended = t > len(run["bodies"]) / run["request"]["physics_hz"]
                caption.set_text(f"{phase}  |  wing {run['angles'][index]:.2f}°" + ("\nControl recording ended; last pose held" if ended else ""))
            cursor.set_xdata([t, t])
            timestamp.set_text(f"{t:05.2f} / {duration:.0f} s · actual speed · pose visualization, not an independent physics replay")
            if frame in (0, 240, 550, frames - 1):
                fig.savefig(output.with_name(output.stem + f"-frame-{frame:04d}.png"))
            writer.grab_frame()
    plt.close(fig)
    check = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-", "-progress", "pipe:1", "-nostats"], capture_output=True, text=True)
    decoded = [int(line.split("=", 1)[1]) for line in check.stdout.splitlines() if line.startswith("frame=")][-1]
    assert check.returncode == 0 and decoded == frames, check.stderr
    report = dict(kind="recorded_native_pose_visualization", full_task_success=False,
                  actual_speed=True, frames=frames, duration_s=duration, decoded_frames=decoded,
                  positive_run=positive.name, negative_run=negative.name,
                  video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plot_bounds_m=dict(lower=lower.tolist(), upper=upper.tolist()),
                  source_sha256={str(run["run"] / "states.npz"): hashlib.sha256((run["run"] / "states.npz").read_bytes()).hexdigest() for run in runs},
                  limitation="Estimated collider geometry; laboratory press and pusher, no robot or cloth. Not new physics or a benchmark demo.")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positive", type=Path)
    parser.add_argument("negative", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(OUTPUT_ROOT)
    render(args.positive, args.negative, args.output)
