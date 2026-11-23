"""Screen rail grasp locations with recorded arm deflection, without physics."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from preflight_cover import KinematicSession
from robot_gate import solve_pose
from robot_reopen import guide_and_place, handle_guidance_target, transform


def preflight(run, out, rail_x_values=(-.025, -.046, -.065),
              centers=((-.1, -.04, 0.),), scales=(0., .8, 1.), measurement='robot'):
    """Compare ordinary command reach under an explicitly constant load model."""
    from station_coordinates import TRANSLATION_S_W
    from yam_arena import ik, station

    request = json.loads((run / 'request.json').read_text())
    shift = TRANSLATION_S_W if request['table_origin'] else np.zeros(3)
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    names = json.loads((run / 'frame-names.json').read_text())
    index = max(i for i, row in enumerate(rows) if row['phase'] == 'verify_inner_rail_grasp')
    row = rows[index]
    physics_index = (index + 1) * (request['physics_hz'] // 20) - 1
    chunk_path = run / 'physics-chunks' / f'{physics_index // request["physics_hz"]:05d}.npz'
    with np.load(chunk_path) as chunk:
        poses = chunk['frame_body_poses'][physics_index % request['physics_hz']]
        joints = chunk['frame_joint_positions'][physics_index % request['physics_hz']]

    def wrist(values, side):
        offset = 0 if side == 'left' else 7
        position, rotation = ik.tcp_pose(np.asarray(values)[offset:offset + 6], station.BASES[side])
        pose = np.eye(4)
        pose[:3, 3], pose[:3, :3] = position + shift, rotation
        return pose

    T_W_B = transform(poses[names['body_names'].index('Bridge')])
    T_W_L = transform(poses[names['body_names'].index('Left')])
    T_W_H = transform(poses[names['body_names'].index('Handle')])
    T_L_TCP = np.linalg.inv(T_W_L) @ wrist(row['measured_joint_pos'], 'left')
    T_H_TCP = np.linalg.inv(T_W_H) @ wrist(row['measured_joint_pos'], 'right')
    T_B_TCP = handle_guidance_target(T_W_B, T_W_H, wrist(row['measured_joint_pos'], 'right'),
                                    float(joints[names['joint_names'].index('HandleY')]))
    angle = -float(joints[names['joint_names'].index('HingeLeft')])
    grasp = json.loads((run / 'rail-grasp-plan.json').read_text())
    last_index = len(rows) * (request['physics_hz'] // 20) - 1
    last_chunk = run / 'physics-chunks' / f'{last_index // request["physics_hz"]:05d}.npz'
    with np.load(last_chunk) as chunk:
        final_poses = chunk['frame_body_poses'][last_index % request['physics_hz']]
    loads = {}
    for side in ('left', 'right'):
        actual = wrist(rows[-1]['measured_joint_pos'], side)
        if measurement == 'body':
            body, local_grasp = ('Left', T_L_TCP) if side == 'left' else ('Handle', T_H_TCP)
            actual = transform(final_poses[names['body_names'].index(body)]) @ local_grasp
        command = wrist(rows[-1]['robot_action'], side)
        loads[side] = (actual[:3, 3] - command[:3, 3],
                      Rotation.from_matrix(actual[:3, :3] @ command[:3, :3].T).as_rotvec())
    results = []
    for rail_x in rail_x_values:
        for scale, center in itertools.product(scales, centers):
            folder = out / f'candidate-{len(results):02d}'
            folder.mkdir(parents=True, exist_ok=False)
            local_grasp = T_L_TCP.copy()
            local_grasp[0, 3] += rail_x - grasp['rail_x_m']
            pose = T_W_L @ local_grasp
            action = np.array(row['robot_action'])
            action[:6], _, _ = solve_pose(pose[:3, 3] - shift, pose[:3, :3], action[:6], 'left')
            session = KinematicSession(action, shift, folder, np.array(center))

            def measure(side):
                value = wrist(session.action, side)
                translation, rotation = loads[side]
                value[:3, 3] += scale * translation
                value[:3, :3] = Rotation.from_rotvec(scale * rotation).as_matrix() @ value[:3, :3]
                return value

            result = dict(rail_x_m=rail_x, load_scale=scale, cover_center_m=list(center))
            try:
                guide_and_place(session, T_W_B, local_grasp, T_B_TCP, angle, grasp['rail_y_m'],
                                .4, shift, measure_wrist=measure if measurement=='robot' else None,
                                measure_grasp=measure if measurement=='body' else None,
                                handle_grasp=T_H_TCP)
                result['passed'] = True
            except AssertionError as error:
                result.update(passed=False, error=str(error), phase=session.phase)
            result['control_steps'] = session.steps + index + 1
            results.append(result)
            print(json.dumps(result), flush=True)
    source_files = ('preflight_loaded_cover.py', 'preflight_cover.py', 'robot_reopen.py',
                    'robot_session.py', 'wrist_tracking.py', 'bridge_press.py')
    report = dict(source_run=run.name, results=results, measurement=measurement,
        scope='CPU command reach/speed under constant measured load deflection. Alternate grasps are hypothetical; no contact, grasp stability or native assembly feasibility is certified.',
        constant_world_load_errors={side: [value.tolist() for value in values] for side, values in loads.items()},
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in source_files},
        recording_sha256={str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (chunk_path, last_chunk, run / 'steps.jsonl', run / 'rail-grasp-plan.json')})
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--rail-x', type=float, nargs='+', default=[-.025, -.046, -.065])
    parser.add_argument('--center', type=float, nargs=3, action='append')
    parser.add_argument('--scales', type=float, nargs='+', default=[0., .8, 1.])
    parser.add_argument('--measurement',choices=('robot','body'),default='robot')
    args = parser.parse_args()
    root = OUTPUT_ROOT
    assert args.run.resolve().parent == root and args.out.resolve().parent == root
    preflight(args.run, args.out, args.rail_x, args.center or [[-.1, -.04, 0.]], args.scales,args.measurement)
