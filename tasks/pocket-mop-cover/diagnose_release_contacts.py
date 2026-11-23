"""Inspect measured arm/frame contact candidates during a selected release interval."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_frame_robot import compare, prepare
from outcome_native import shape_clearance
from preflight_supported_release import projection_shape
from preflight_upright_release import moved
from robot_reopen import transform


def diagnose(run, side, start, end, native_substeps=False):
    """Check selected measured poses; this does not measure contact forces."""
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    hz = json.loads((run / 'request.json').read_text())['physics_hz']
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    robots = json.loads((run / 'robot-cooked-colliders.json').read_text())[side]
    geometry = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    identity = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    frame = [prepare(shape, identity) for shape in geometry if shape['enabled']]
    arm = [prepare(shape, identity) for shape in robots['colliders']]
    hashes = {}
    samples = []
    chunk_index = None
    native = None
    decimation = hz // 20
    assert 0. <= start <= end <= rows[-1]['time_s']
    if native_substeps:
        indices = range(max(0, math.ceil(start * hz - 1e-8) - 1), math.floor(end * hz + 1e-8))
    else:
        indices = [(row['step'] + 1) * decimation - 1 for row in rows if start <= row['time_s'] <= end]
    for index in indices:
        row = rows[index // decimation]
        if index // hz != chunk_index:
            chunk_index = index // hz
            path = run / 'physics-chunks' / f'{chunk_index:05d}.npz'
            hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as chunk:
                native = {key: chunk[key] for key in ('frame_body_poses', side + '_body_poses')}
        poses = native['frame_body_poses'][index % hz]
        links = native[side + '_body_poses'][index % hz]
        hardware = [moved(shape, transform(poses[names.index(shape['body'])])) for shape in frame]
        hand = [moved(shape, transform(links[robots['body_names'].index(shape['body'])])) for shape in arm]
        near = []
        for first in hardware:
            for second in hand:
                gap = np.maximum(first['global_low'] - second['global_high'],
                                 second['global_low'] - first['global_high'])
                if np.linalg.norm(np.maximum(gap, 0.)) > .002:
                    continue
                bound = shape_clearance([projection_shape(first)], [projection_shape(second)], .002)
                if bound > .002:
                    continue
                depth, _, _ = compare(first, second)
                near.append(dict(frame=first['body'] + '/' + first['name'],
                    robot=second['name'], clearance_lower_bound_m=bound, penetration_upper_bound_m=depth))
        bridge = poses[names.index('Bridge')]
        samples.append(dict(time_s=(index + 1) / hz, phase=row['phase'],
            bridge_rpy_deg=Rotation.from_quat(bridge[3:]).as_euler('xyz', degrees=True).tolist(),
            bridge_height_m=float(bridge[2]), contacts=near))
    for filename in ('steps.jsonl', 'request.json', 'frame-names.json', 'frame-geometry.json',
                     'robot-cooked-colliders.json', 'manifest.json'):
        hashes[filename] = hashlib.sha256((run / filename).read_bytes()).hexdigest()
    report = dict(source_run=run.name, moving_arm=side, interval_s=[start, end], samples=samples,
        sampling_hz=hz if native_substeps else 20,
        scope='Selected measured poses and conservative geometry bounds only. Positive penetration bounds or zero separation identify possible contacts, not contact forces. Only the named arm and selected interval are checked; a complete independent episode audit remains necessary.',
        source_sha256=hashes, diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    suffix = '-release-native-contact-diagnostic.json' if native_substeps else '-release-contact-diagnostic.json'
    (run / (side + suffix)).write_text(json.dumps(report, indent=2))
    print(json.dumps(samples, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--side', choices=('left', 'right'), required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--native-substeps', action='store_true')
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    diagnose(args.run, args.side, args.start, args.end, args.native_substeps)
