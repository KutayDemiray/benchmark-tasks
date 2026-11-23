"""Inspect retention geometry without issuing a numerical or benchmark certificate."""

from task_paths import OUTPUT_ROOT

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from outcome_native import NativeOutcome


def diagnose(run, start):
    """Record geometric observations while deliberately disabling score validity."""
    request = json.loads((run / 'request.json').read_text())
    scene = json.loads((run / 'scene-audit.json').read_text())
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    chunks = sorted((run / 'physics-chunks').glob('[0-9][0-9][0-9][0-9][0-9].npz'))
    completed = (run / 'result.json').exists()
    count = len(rows) if completed else min(len(rows), max(0, len(chunks) - 1) * 20)
    hz = request['physics_hz']
    decimation = hz // 20
    first = round(start * 20)
    assert first < count and first % 20 == 0
    geometry = np.load(run / 'geometry.npz')
    center = np.asarray(scene['table_center_m'][:2])
    half = np.asarray(scene['table_size_m'][:2]) / 2
    observer = NativeOutcome(
        geometry['initial'], geometry['faces'], json.loads((run / 'cover-mesh.json').read_text()),
        json.loads((run / 'frame-geometry.json').read_text()),
        json.loads((run / 'frame-names.json').read_text()),
        json.loads((run / 'robot-cooked-colliders.json').read_text()),
        scene['table_center_m'][2] + scene['table_size_m'][2] / 2, hz,
        # Unknown validity is deliberately false, so this diagnostic cannot
        # produce a positive score. Geometry/history still use the real reducer.
        dict(finite=False), (center - half, center + half))
    observer.scorer.last_step = first - 1
    if first:
        with np.load(chunks[first // 20 - 1]) as previous:
            observer.previous = previous['positions'][-1].copy()
    phases = {}
    first_lift = None
    maximum_distance = 0.
    any_retention = False
    fields = ('positions', 'velocities', 'frame_body_poses', 'frame_joint_positions',
              'left_body_poses', 'right_body_poses')
    output = run / 'retention-geometry-observations.jsonl'
    with output.open('w') as stream:
        for chunk_index in range(first // 20, (count + 19) // 20):
            with np.load(chunks[chunk_index]) as chunk:
                recorded = {name: chunk[name] for name in fields}
            for local in range(min(20, count - chunk_index * 20)):
                index = chunk_index * 20 + local
                offset = local * decimation
                state = {name: values[offset:offset + decimation] for name, values in recorded.items()}
                observation, result = observer.update(index, state)
                assert result['score'] == 0 and not result['success']
                reference = observer.scorer.lift_xy
                distance = None if reference is None else float(np.linalg.norm(
                    np.asarray(observation.head_xy_m) - reference))
                entry = dict(time_s=rows[index]['time_s'], phase=rows[index]['phase'],
                    observation=asdict(observation), lift_reference_xy_m=reference,
                    distance_from_reference_m=distance,
                    retention_geometry_verified=result['retention_verified'],
                    release_geometry_hold_seconds=result['released_hold_seconds'])
                if reference is not None and first_lift is None:
                    first_lift = dict(time_s=entry['time_s'], xy_m=reference)
                maximum_distance = max(maximum_distance, distance or 0.)
                any_retention |= result['retention_verified']
                phases[entry['phase']] = entry
                stream.write(json.dumps(entry) + '\n')
    report = dict(scope='Geometry-only diagnostic; numerical validity deliberately disabled. No benchmark score.',
        start_s=start, observed_until_s=count / 20, first_lift=first_lift,
        maximum_distance_from_reference_m=maximum_distance,
        any_retention_geometry_verified=any_retention, phase_end_observations=list(phases.values()))
    (run / 'retention-geometry-diagnostic.json').write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key != 'phase_end_observations'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--start', type=int, default=70)
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    print(json.dumps(diagnose(args.run, args.start), indent=2))
