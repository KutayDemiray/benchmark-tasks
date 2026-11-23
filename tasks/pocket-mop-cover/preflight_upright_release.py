"""Screen the final placement path against recorded and predicted rigid geometry."""

from task_paths import OUTPUT_ROOT

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_frame_robot import compare, prepare
from preflight_cover import KinematicSession
from robot_clearance import link_transforms
from robot_reopen import transform
from upright_release import place_upright, upright_target
from outcome_native import gap


def moved(source, pose):
    """Transform cached convex features without recomputing their hull."""
    _, _, vertices, normals, edges = source['data']
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    low, high = world.min(0), world.max(0)
    return {**source, 'data': (pose[None, :3, 3], pose[None, :3, :3], vertices, normals, edges),
        'low': low[None], 'high': high[None], 'global_low': low, 'global_high': high}


def preflight(run, out):
    """Check ordinary reach plus a disclosed rigid planning approximation."""
    from station_coordinates import TRANSLATION_S_W
    from yam_arena import ik, station

    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    index = max(i for i, row in enumerate(rows) if row['phase'] == 'retention')
    row = rows[index]
    hz = json.loads((run / 'request.json').read_text())['physics_hz']
    native_index = (index + 1) * (hz // 20) - 1
    with np.load(run / 'physics-chunks' / f'{native_index // hz:05d}.npz') as chunk:
        poses = chunk['frame_body_poses'][native_index % hz]
        left_poses = chunk['left_body_poses'][native_index % hz]
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    robots = json.loads((run / 'robot-cooked-colliders.json').read_text())
    shapes = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    scene = json.loads((run / 'scene-audit.json').read_text())
    bridge, handle = (transform(row[key]) for key in ('bridge_pose_xyzw', 'handle_pose_xyzw'))
    position, rotation = ik.tcp_pose(np.array(row['measured_joint_pos'])[7:13], station.BASES['right'])
    wrist = np.eye(4)
    wrist[:3, 3], wrist[:3, :3] = position + TRANSLATION_S_W, rotation
    local_grasp = np.linalg.inv(handle) @ wrist
    upright = upright_target(bridge, handle, wrist)

    class RecordedKinematics(KinematicSession):
        def step(self, command, phase):
            super().step(command, phase)
            self.samples.append((command.copy(), phase))

    session = RecordedKinematics(np.array(row['robot_action']), TRANSLATION_S_W, out, None)
    session.samples = []
    place_upright(session, bridge, handle, wrist, 0.)
    identity = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    right = [prepare(shape, identity) for shape in robots['right']['colliders']]
    left = [prepare(shape, left_poses[None, robots['left']['body_names'].index(shape['body'])])
            for shape in robots['left']['colliders']]
    frame = [prepare(shape, identity) for shape in shapes if shape['enabled']]
    relative = {name: np.linalg.inv(bridge) @ transform(poses[names.index(name)]) for name in names}
    table_points = np.array(list(itertools.product((-1., 1.), repeat=3))) * np.asarray(scene['table_size_m']) / 2
    table = prepare(dict(name='Table', body='Table', points=table_points),
        np.array([[*scene['table_center_m'], 0., 0., 0., 1.]]))
    checked, hits, seen = 0, [], set()
    wing_gaps = []
    final_hand_gap = None
    placed_bridge = bridge.copy()
    for command, phase in session.samples:
        key = command.tobytes()
        if key in seen:
            continue
        seen.add(key)
        links = link_transforms(command[7:], 'right', TRANSLATION_S_W)
        arm = [moved(shape, links[shape['body']]) for shape in right]
        point, rotation = ik.tcp_pose(command[7:13], station.BASES['right'])
        tcp = np.eye(4)
        tcp[:3, 3], tcp[:3, :3] = point + TRANSLATION_S_W, rotation
        predicted_bridge = bridge.copy()
        if phase in ('set_down_frame', 'level_handle_before_release'):
            predicted_bridge[:3, 3] += tcp[:3, 3] - upright[:3, 3]
            placed_bridge = predicted_bridge.copy()
        elif phase in ('release_handle', 'clear_upright_shaft', 'withdraw', 'released'):
            predicted_bridge = placed_bridge
        predicted_handle = tcp @ np.linalg.inv(local_grasp)
        if phase == 'release_handle':
            released_handle = predicted_handle.copy()
        if phase in ('clear_upright_shaft', 'withdraw', 'released'):
            predicted_handle = released_handle
        hardware = [moved(shape, predicted_handle if shape['body'] == 'Handle'
                          else predicted_bridge @ relative[shape['body']]) for shape in frame]
        wing_gap = min(float(gap(a['global_low'], a['global_high'], b['global_low'], b['global_high']))
            for a in hardware if a['body'] in ('Left', 'Right') for b in arm + left)
        wing_gaps.append(dict(sample=checked, phase=phase, lower_bound_m=wing_gap))
        final_hand_gap = min(float(gap(a['global_low'], a['global_high'], b['global_low'], b['global_high']))
            for a in hardware for b in arm + left if b['body'] in ('gripper', 'tip_left', 'tip_right'))
        for first, second in ((arm, left + [table] + hardware), (left, hardware)):
            low_a, high_a = (np.array([shape[key] for shape in first]) for key in ('global_low', 'global_high'))
            low_b, high_b = (np.array([shape[key] for shape in second]) for key in ('global_low', 'global_high'))
            overlap = np.all(high_a[:, None] >= low_b, axis=2) & np.all(high_b >= low_a[:, None], axis=2)
            for a, b in np.argwhere(overlap):
                one, two = first[a], second[b]
                if one['body'] == 'base' and two['body'] == 'Table':
                    continue
                # A commanded closed gap is smaller than the held shaft;
                # native contact must determine actual finger separation.
                if (phase not in ('clear_upright_shaft', 'withdraw', 'released')
                        and two['body'] == 'Handle' and one['body'] in ('tip_left', 'tip_right')):
                    continue
                depth, _, _ = compare(one, two)
                if depth > .001:
                    hits.append(dict(sample=checked, phase=phase, first=one['name'],
                        second=two['body'] + '/' + two['name'], depth_bound_m=depth))
        checked += 1
    report = dict(passed=not hits, control_steps=session.steps, unique_poses=checked, hits=hits,
        minimum_predicted_wing_robot_gap_m=min(entry['lower_bound_m'] for entry in wing_gaps),
        possible_wing_support=[entry for entry in wing_gaps if entry['lower_bound_m'] < .002],
        final_predicted_frame_hand_gap_m=final_hand_gap,
        scope='Kinematic planning approximation: measured parked left arm, command right arm, transported frame and held shaft. Does not certify native feasibility; finger/shaft grasp contact is excluded.')
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    assert args.run.resolve().parent == args.out.resolve().parent == OUTPUT_ROOT
    args.out.mkdir(exist_ok=False)
    preflight(args.run, args.out)
