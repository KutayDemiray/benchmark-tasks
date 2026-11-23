"""Plot recorded peel replay divergence without launching simulation."""

from task_paths import OUTPUT_ROOT

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    source = np.load(args.source / 'per-step.npz')
    replay = np.load(args.replay / 'per-step.npz')
    a, b = source['male_poses'], replay['male_poses']
    assert a.shape == b.shape
    time = np.arange(len(a)) * float(source['dt'])
    position_error = np.max(np.abs(a[:, :3] - b[:, :3]), axis=1)
    dot = np.sum(a[:, 3:] * b[:, 3:], axis=1)
    angle_error = 2*np.arccos(np.clip(np.abs(dot), 0, 1))
    ca = np.load(args.source / 'upper-nodes.npz')['positions']
    cb = np.load(args.replay / 'upper-nodes.npz')['positions']
    cloth_error = np.max(np.abs(ca-cb), axis=(1, 2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, 1000*position_error, color='#315f8c')
    axes[0].axhline(1., color='black', linestyle='--', linewidth=.8)
    axes[0].set_ylabel('Snap coordinate error (mm)')
    axes[1].plot(time, np.rad2deg(angle_error), color='#a45031')
    axes[1].axhline(5., color='black', linestyle='--', linewidth=.8)
    axes[1].set_ylabel('Snap orientation error (degrees)')
    axes[2].plot(time, 1000*cloth_error, color='#3b7b60')
    axes[2].axhline(1., color='black', linestyle='--', linewidth=.8)
    axes[2].set_ylabel('Upper cloth coordinate error (mm)')
    axes[2].set_xlabel('Simulation time (s)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.axvline(3.3, color='gray', linestyle=':', linewidth=1.)
    fig.suptitle('Unguided peel: identical recorded commands, independent native process\n'
                 'Dashed lines: frozen replay tolerances; vertical line: peel begins')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
