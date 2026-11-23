"""Screen supported release against cooked geometry using an explicit planning model."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_frame_robot import compare, prepare
from outcome_native import shape_clearance
from preflight_cover import KinematicSession
from preflight_upright_release import moved
from robot_clearance import link_transforms
from robot_reopen import handle_pose, transform
from supported_release import PLACEMENT_OFFSET, handle_angles, place_supported


def projection_shape(shape):
    """Expose conservative projection features for a planned rigid pose."""
    position, rotation, vertices, normals, _ = shape['data']
    return dict(body=shape['body'], low=shape['global_low'], high=shape['global_high'],
        cloud=vertices @ rotation[0].T + position[0], axes=normals @ rotation[0].T)


def preflight(run, out, fallen_handle=False):
    """Check a recorded placement followed by commanded two-hand release."""
    from station_coordinates import TRANSLATION_S_W
    from yam_arena import ik, station

    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    index = max(i for i, row in enumerate(rows) if row['phase'] == 'set_down_frame')
    row = rows[index]
    hz = json.loads((run / 'request.json').read_text())['physics_hz']
    native_index = (index + 1) * (hz // 20) - 1
    native_file = run / 'physics-chunks' / f'{native_index // hz:05d}.npz'
    with np.load(native_file) as chunk:
        poses = chunk['frame_body_poses'][native_index % hz]
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    bridge, handle = (transform(row[key]) for key in ('bridge_pose_xyzw', 'handle_pose_xyzw'))

    class PlanningSession(KinematicSession):
        def __init__(self):
            super().__init__(np.array(row['robot_action']), TRANSLATION_S_W, out, None)
            self.bridge, self.handle = bridge.copy(), handle.copy()
            self.local_grasp = None
            self.bridge_grasp = None
            self.released_bridge_grasp = None
            self.samples = []

        def wrist(self, side):
            offset = 0 if side == 'left' else 7
            point, rotation = ik.tcp_pose(self.action[offset:offset + 6], station.BASES[side])
            value = np.eye(4)
            value[:3, 3], value[:3, :3] = point + self.shift, rotation
            return value

        def step(self, command, phase):
            if phase == 'level_grasped_bridge' and self.bridge_grasp is None:
                self.bridge_grasp = np.linalg.inv(self.bridge) @ self.wrist('left')
                self.bridge_handle = np.linalg.inv(self.bridge) @ self.handle
            if phase == 'lower_supported_handle' and self.local_grasp is None:
                self.local_grasp = np.linalg.inv(self.handle) @ self.wrist('right')
            if phase == 'level_released_bridge' and self.released_bridge_grasp is None:
                self.released_bridge_grasp = np.linalg.inv(self.bridge) @ self.wrist('left')
                self.released_bridge_handle = np.linalg.inv(self.bridge) @ self.handle
            super().step(command, phase)
            if phase == 'set_down_frame':
                offset = ik.smoothstep(self.steps / 60) * PLACEMENT_OFFSET
                self.bridge[:3, 3] = bridge[:3, 3] + offset
                self.handle[:3, 3] = handle[:3, 3] + offset
            elif phase == 'level_grasped_bridge':
                self.bridge = self.wrist('left') @ np.linalg.inv(self.bridge_grasp)
                self.handle = self.bridge @ self.bridge_handle
            elif phase in ('lower_supported_handle', 'align_supported_handle'):
                self.handle = self.wrist('right') @ np.linalg.inv(self.local_grasp)
            elif phase == 'level_released_bridge':
                self.bridge = self.wrist('left') @ np.linalg.inv(self.released_bridge_grasp)
                self.handle = self.bridge @ self.released_bridge_handle
            self.samples.append((command.copy(), phase, self.bridge.copy(), self.handle.copy()))

    session = PlanningSession()
    place = session.wrist('right')
    place_supported(session, place[:3, 3], place[:3, :3], bridge, handle, place, 0.,
                    lambda: {'Bridge': session.bridge, 'Handle': session.handle}, session.wrist)
    robots = json.loads((run / 'robot-cooked-colliders.json').read_text())
    shapes = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    scene = json.loads((run / 'scene-audit.json').read_text())
    identity = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    robot_shapes = {side: [prepare(shape, identity) for shape in robots[side]['colliders']]
                    for side in ('left', 'right')}
    frame_shapes = [prepare(shape, identity) for shape in shapes if shape['enabled']]
    relative = {name: np.linalg.inv(bridge) @ transform(poses[names.index(name)]) for name in names}
    table_points = np.array(list(itertools.product((-1., 1.), repeat=3))) * np.asarray(scene['table_size_m']) / 2
    table = prepare(dict(name='Table', body='Table', points=table_points),
        np.array([[*scene['table_center_m'], 0., 0., 0., 1.]]))
    hits, seen, supports = [], set(), []
    minimum_wing_gap = float('inf')
    final_hand_gap = None
    for sample, (command, phase, current_bridge, current_handle) in enumerate(session.samples):
        key = command.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if fallen_handle and command[13] >= .24 and phase in ('clear_supported_handle', 'withdraw_released_handle',
                                       'clear_release_support', 'withdraw_release_support', 'released'):
            current_handle = handle_pose(current_bridge, 0., np.deg2rad(85.))
        arms = {}
        for side, offset in (('left', 0), ('right', 7)):
            links = link_transforms(command[offset:offset + 7], side, TRANSLATION_S_W)
            arms[side] = [moved(shape, links[shape['body']]) for shape in robot_shapes[side]]
        hardware = []
        x, _ = handle_angles(current_bridge, current_handle)
        connector = current_bridge.copy()
        connector[:3, 3] += current_bridge[:3, :3] @ [.030, 0., .052]
        connector[:3, :3] = current_bridge[:3, :3] @ Rotation.from_euler('x', x).as_matrix()
        for shape in frame_shapes:
            pose = current_bridge @ relative[shape['body']]
            if shape['body'] == 'Handle':
                pose = current_handle
            elif shape['body'] == 'Connector':
                pose = connector
            hardware.append(moved(shape, pose))
        projected_hardware = [projection_shape(shape) for shape in hardware]
        projected_robots = [projection_shape(shape) for arm in arms.values() for shape in arm]
        wing_gap = shape_clearance([shape for shape in projected_hardware
                                   if shape['body'] in ('Left', 'Right')], projected_robots, .002)
        minimum_wing_gap = min(minimum_wing_gap, wing_gap)
        if wing_gap < .002:
            supports.append(dict(sample=sample, phase=phase, separation_lower_bound_m=wing_gap))
        final_hand_gap = shape_clearance(projected_hardware,
            [shape for shape in projected_robots if shape['body'] in ('gripper', 'tip_left', 'tip_right')])
        for side, first, second in (('right', arms['right'], arms['left'] + [table] + hardware),
                                    ('left', arms['left'], [table] + hardware)):
            low_a, high_a = (np.array([shape[key] for shape in first]) for key in ('global_low', 'global_high'))
            low_b, high_b = (np.array([shape[key] for shape in second]) for key in ('global_low', 'global_high'))
            overlap = np.all(high_a[:, None] >= low_b, axis=2) & np.all(high_b >= low_a[:, None], axis=2)
            for a, b in np.argwhere(overlap):
                one, two = first[a], second[b]
                if one['body'] == 'base' and two['body'] == 'Table':
                    continue
                if one['body'] in ('tip_left', 'tip_right'):
                    if (side == 'right' and two['body'] == 'Handle'
                            and (command[13] < .24 or phase == 'open_supported_handle')):
                        # Commanded jaw positions continue through the held
                        # shaft until the outward opening clears it. Native
                        # contact determines their actual separation. Check
                        # every shaft/finger pair once wrist withdrawal starts.
                        continue
                    if side == 'left' and two['body'] == 'Bridge' and phase in (
                        'grasp_release_bridge', 'level_grasped_bridge', 'lower_supported_handle', 'align_supported_handle',
                        'hold_release_support', 'open_supported_handle', 'clear_supported_handle',
                        'withdraw_released_handle', 'level_released_bridge', 'hold_releveled_bridge',
                        'open_release_bridge'):
                        continue
                depth, _, _ = compare(one, two)
                if depth > .001:
                    hits.append(dict(sample=sample, phase=phase, side=side, first=one['name'],
                                     second=two['body'] + '/' + two['name'], depth_bound_m=depth))
    report = dict(passed=not hits and not supports and final_hand_gap >= .02,
        control_steps=session.steps, unique_poses=len(seen), hits=hits,
        minimum_predicted_wing_robot_gap_m=minimum_wing_gap, possible_wing_support=supports,
        final_predicted_frame_hand_gap_m=final_hand_gap,
        source_run=run.name, released_handle_at_85_deg=fallen_handle,
        scope='Planning only: starts from the recorded set-down head, assumes the commanded left bridge grasp holds during leveling, and transports the handle with the right command grasp. Intentional finger/shaft and finger/bridge grasp overlap is excluded until withdrawal. No cloth, dynamic clearance, contact force, native grasp or release stability is certified.',
        input_sha256={str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (run / 'steps.jsonl', native_file, run / 'robot-cooked-colliders.json')},
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('preflight_supported_release.py', 'supported_release.py', 'robot_session.py')})
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'hits'}))
    print('potential_collisions', len(hits), hits[:5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--fallen-handle', action='store_true')
    args = parser.parse_args()
    root = OUTPUT_ROOT
    assert args.run.resolve().parent == args.out.resolve().parent == root
    args.out.mkdir(exist_ok=False)
    preflight(args.run, args.out, args.fallen_handle)
