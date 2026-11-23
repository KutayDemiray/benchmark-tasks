"""Render measured diagnostic plots on CPU; these are not camera videos."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (args.episode / 'trace.jsonl').read_text().splitlines()]
    t = np.array([r['time_s'] for r in records])
    height = np.array([r['male_pose_xyzw'][2] - .1 for r in records]) * 1000
    force = np.array([r['diagnostic_force_N'][2] for r in records])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, height, color='#315f8c')
    axes[0].set_ylabel('Cap height above socket base (mm)')
    axes[1].plot(t, force, color='#b24b34')
    axes[1].set_ylabel('Diagnostic axial force (N)')
    for name in ('upper', 'lower'):
        if name in records[0]['cloth']:
            axes[2].plot(t, [r['cloth'][name]['max_abs_edge_strain'] for r in records], label=name)
    axes[2].axhline(.15, color='black', linestyle='--', linewidth=.8, label='provisional strain limit')
    axes[2].set_ylabel('Maximum absolute edge strain')
    axes[2].set_xlabel('Simulation time (s)')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
        for start in (.3, 2.3, 2.8, 3.3):
            ax.axvline(start, color='gray', linestyle=':', linewidth=.7)
    fig.suptitle('Isolated snap fixture — measured simulation, not a robot demonstration')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
