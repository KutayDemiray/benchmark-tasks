"""Measure contact-grasp drift during recorded final handle repositioning."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_reopen import transform


def diagnose(run):
    """Report measured relative wrist motion without certifying a physical grasp."""
    from station_coordinates import TRANSLATION_S_W
    from yam_arena import ik, station
    from yam_arena.constants import FINGER_OPEN

    request = json.loads((run / 'request.json').read_text())
    shift = TRANSLATION_S_W if request['table_origin'] else np.zeros(3)
    recording = (run / 'steps.jsonl').read_text()
    if not recording.endswith('\n'):
        recording = recording.rsplit('\n', 1)[0] + '\n'
    lines = recording.splitlines()
    rows = [json.loads(line) for line in lines]
    first = next((i for i, row in enumerate(rows) if row['phase'] in (
        'grasp_release_bridge', 'steady_release_bridge')), None)
    if first is None:
        print('Final bridge contact has not started.')
        return
    baseline_index = max(i for i, row in enumerate(rows)
                         if row['phase'] == rows[first]['phase'])
    selected = {row['phase']: index for index, row in enumerate(rows) if index >= first}
    baseline = None
    reports = []
    for phase, index in selected.items():
        row = rows[index]
        poses = {}
        wrists = {}
        for side, offset, owner in (('left', 0, 'bridge_pose_xyzw'),
                                    ('right', 7, 'handle_pose_xyzw')):
            point, rotation = ik.tcp_pose(np.array(row['measured_joint_pos'])[offset:offset + 6],
                                          station.BASES[side])
            wrist = np.eye(4)
            wrist[:3, 3], wrist[:3, :3] = point + shift, rotation
            wrists[side] = wrist
            poses[side] = np.linalg.inv(transform(row[owner])) @ wrist
        if index == baseline_index:
            baseline = poses
        drift = {}
        if baseline is not None:
            for side in ('left', 'right'):
                drift[side] = dict(
                    translation_m=float(np.linalg.norm(poses[side][:3, 3] - baseline[side][:3, 3])),
                    rotation_deg=float(np.rad2deg(Rotation.from_matrix(
                        poses[side][:3, :3] @ baseline[side][:3, :3].T).magnitude())))
        reports.append(dict(phase=phase, step=index, time_s=row['time_s'],
            left_nominal_jaw_gap_m=float(2 * abs(FINGER_OPEN) * row['measured_joint_pos'][6]),
            left_commanded_opening=row['robot_action'][6],
            left_bridge_relative_tcp_m=poses['left'][:3, 3].tolist(),
            right_handle_relative_tcp_m=poses['right'][:3, 3].tolist(),
            drift_from_first_contact=drift,
            bridge_rpy_deg=Rotation.from_quat(row['bridge_pose_xyzw'][3:]).as_euler('xyz', degrees=True).tolist(),
            bridge_height_m=row['bridge_pose_xyzw'][2],
            handle_gimbal_deg={name: value for name, value in row['frame_joint_angles_deg'].items()
                                if name.startswith('Handle')}, pockets=row['pockets']))
    report = dict(scope='Read-only pose and aperture diagnosis. Nominal jaw gap comes from the stock normalized joint state, not minimum mesh-surface distance. Relative stability does not certify contact force, friction or numerical validity. Phase names select diagnostic samples only.',
        source_control_steps=len(rows), source_recording_snapshot_sha256=hashlib.sha256(recording.encode()).hexdigest(),
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), samples=reports)
    (run / 'bridge-grasp-diagnostic.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    diagnose(args.run)
