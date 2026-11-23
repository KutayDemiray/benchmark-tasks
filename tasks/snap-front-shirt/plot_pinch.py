"""Plot native pinch measurements and the zero-friction control on CPU."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for path in sorted(args.run.glob('episode-*')):
        setup = json.loads((path / 'setup.json').read_text())
        if setup['case'] not in ('grasp', 'frictionless'):
            continue
        data = np.load(path / 'per-step.npz')
        t = np.arange(len(data['male_poses'])) * float(data['dt'])
        ax.plot(t, 1000*(data['male_poses'][:, 2]-.1), label=setup['case'])
    ax.axhline(15, color='gray', linestyle=':', label='15 mm lift check')
    ax.axvline(2.2, color='gray', linestyle='--', label='jaws begin opening')
    ax.set(xlabel='Simulation time (s)', ylabel='Attached cap height relative to reset (mm)',
           title='Frictional cloth-pinch fixture — not a YAM demonstration')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)


if __name__ == '__main__':
    main()
