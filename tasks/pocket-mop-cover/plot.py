"""Render diagnostic figures from measured native nodes, without simulation."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.is_relative_to(OUTPUT_ROOT)
    assert (run / "result.json").exists(), "Recording has not finished; wait for result.json"
    data = np.load(run / "states.npz")
    trace = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    times = [r["time_s"] for r in trace]
    fig = plt.figure(figsize=(13, 8))
    for panel, fraction in enumerate((0, 0.3, 0.6, 1)):
        index = round(fraction * (len(trace) - 1))
        ax = fig.add_subplot(2, 3, panel + 1, projection="3d")
        verts = data["positions"][index]
        coll = Poly3DCollection(verts[data["faces"]], facecolor="#477ea8", edgecolor="#224466", linewidth=0.1, alpha=0.9)
        ax.add_collection3d(coll)
        c = data["blade_positions"][index]
        corners = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.003, 0.003)]) + c
        indices = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        ax.add_collection3d(Poly3DCollection(corners[indices], facecolor="#d4b64b", alpha=0.6))
        ax.set(xlim=(-0.03, 0.27), ylim=(-0.11, 0.11), zlim=(-0.01, 0.09), title=f"Measured state at {times[index]:.3f} s", xlabel="X (m)", ylabel="Y (m)")
        ax.set_box_aspect((3, 2.2, 1))
        ax.view_init(elev=20, azim=-58)
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(times, [r["max_abs_edge_strain"] for r in trace], label="Maximum absolute edge strain")
    ax.axhline(0.2, color="red", linestyle="--", label="Invalid threshold")
    ax.set(xlabel="Simulated seconds", ylabel="Relative strain")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(times, 1000 * data["blade_positions"][:, 0], label="Measured blade X")
    ax.plot(times, [1000 * r["rig_command_m"][0] for r in trace], label="Commanded blade X", linestyle="--")
    ax.set(xlabel="Simulated seconds", ylabel="Position (mm)")
    ax.legend(fontsize=7)
    fig.suptitle("Isolated pocket diagnostic — measured states, not a robot demonstration")
    fig.tight_layout()
    fig.savefig(run / "diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
