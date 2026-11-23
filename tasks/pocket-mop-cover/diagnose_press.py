"""Compare recorded wrist commands, arm deflection and passive latch motion."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def diagnose(run, side='right'):
    """Save a closing-trajectory diagnostic without issuing a physics certificate."""
    from yam_arena import ik, station
    from yam_arena.constants import SHOULDER, WRIST

    from station_coordinates import TRANSLATION_S_W

    request = json.loads((run / 'request.json').read_text())
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    shift = TRANSLATION_S_W if request['table_origin'] else np.zeros(3)
    stiffness = np.array([SHOULDER['stiffness']] * 3 + [WRIST['stiffness']] * 3)
    damping = np.array([SHOULDER['damping']] * 3 + [WRIST['damping']] * 3)
    assert side in ('left', 'right')
    joint_start = 0 if side == 'left' else 7
    phases = {'press_frame_flat', 'finish_frame_press', 'hold_pressed_frame'}
    if side == 'left':
        phases = {'touch_bridge_rail', 'press_bridge_rail', 'hold_bridge_press', 'clear_bridge_press'}
    records = []
    for index, row in enumerate(rows):
        if row['phase'] not in phases:
            continue
        measured = np.asarray(row['measured_joint_pos'])[joint_start:joint_start + 6]
        command = np.asarray(row['robot_action'])[joint_start:joint_start + 6]
        position, rotation = ik.tcp_pose(measured, station.BASES[side])
        target, _ = ik.tcp_pose(command, station.BASES[side])
        previous = np.asarray(rows[max(index - 1, 0)]['measured_joint_pos'])[joint_start:joint_start + 6]
        velocity = (measured - previous) * 20
        jacobian = np.zeros((6, 6))
        for joint in range(6):
            offset = measured.copy()
            offset[joint] += 1e-6
            point, orientation = ik.tcp_pose(offset, station.BASES[side])
            jacobian[:3, joint] = (point - position) / 1e-6
            jacobian[3:, joint] = Rotation.from_matrix(orientation @ rotation.T).as_rotvec() / 1e-6
        torque = stiffness * (command - measured) - damping * velocity
        # Stock actions supply arm gravity compensation separately. Remaining
        # PD torque is an approximate load wrench only under quasistatic motion.
        wrench = np.linalg.solve(jacobian.T, torque)
        joint_margin = np.minimum(measured - ik.LOWER, ik.UPPER - measured)
        angles = row['frame_joint_angles_deg']
        records.append(dict(
            time_s=row['time_s'], phase=row['phase'],
            command_wrist_m=(target + shift).tolist(),
            measured_wrist_m=(position + shift).tolist(),
            bridge_m=row['bridge_pose_xyzw'][:3],
            fold_deg=max(abs(value) for value in row['wing_angles_deg']),
            rocker_deg=angles['RockerHinge'], clip_deg=angles['ClipFlexure'],
            handle_x_deg=angles['HandleX'], handle_y_deg=angles['HandleY'],
            nominal_force_robot_on_load_N=wrench[:3].tolist(),
            nominal_torque_robot_on_load_Nm=wrench[3:].tolist(),
            max_joint_speed_rad_s=float(np.max(abs(velocity))),
            minimum_joint_limit_margin_rad=float(joint_margin.min()),
            limiting_joint_index=int(joint_margin.argmin()),
            joint_limit_reaction_possible=bool(joint_margin.min() < 1e-3),
            jacobian_condition=float(np.linalg.cond(jacobian)),
            both_pockets_live_enclosed=row['both_pockets_seated']))
    assert records, 'No recorded closing trajectory'
    report = dict(
        source_run=run.name, arm=side, gravity_compensation_enabled=True,
        scope='Recorded closing diagnostic only. Wrench is an approximate quasistatic PD/Jacobian estimate, not a native force measurement: joint friction, inertia, effort saturation and joint-limit reactions are not resolved. Live pocket checks do not replace independent material/outcome audits.',
        source_sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                       for name in ('steps.jsonl', 'request.json')},
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=records)
    filename = 'closing-diagnostic.json' if side == 'right' else 'closing-diagnostic-left.json'
    (run / filename).write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--side', choices=('left', 'right'), default='right')
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    result = diagnose(args.run, args.side)
    print(json.dumps(dict(source_run=result['source_run'], samples=len(result['records']),
                          final=result['records'][-1]), indent=2))
