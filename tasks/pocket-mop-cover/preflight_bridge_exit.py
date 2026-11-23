"""Screen either hand's withdrawal from an actual native release state."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_frame_robot import compare, prepare
from outcome_native import cloth_clearance, shape_clearance
from preflight_cover import KinematicSession
from preflight_supported_release import projection_shape
from preflight_upright_release import moved
from robot_clearance import link_transforms
from robot_reopen import handle_pose, transform
from supported_release import level_bridge_target, release_bridge, release_handle


def preflight(run, out, source_phase='hold_release_support', side='left', released_handle=False,
              level_before_exit=False, preserve_held_aperture=False, unload_during_opening=False):
    """Check command and recorded-deflection models without running physics."""
    from station_coordinates import TRANSLATION_S_W
    from yam_arena import ik, station

    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    index = max(i for i, row in enumerate(rows) if row['phase'] == source_phase)
    row = rows[index]
    hz = json.loads((run / 'request.json').read_text())['physics_hz']
    native_index = (index + 1) * (hz // 20) - 1
    native_file = run / 'physics-chunks' / f'{native_index // hz:05d}.npz'
    with np.load(native_file) as chunk:
        native = {key: chunk[key][native_index % hz].copy() for key in
                  ('frame_body_poses', 'left_body_poses', 'right_body_poses', 'positions')}
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    robots = json.loads((run / 'robot-cooked-colliders.json').read_text())
    frame = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    scene = json.loads((run / 'scene-audit.json').read_text())
    with np.load(run / 'geometry.npz') as mesh:
        faces = mesh['faces']

    class ExitSession(KinematicSession):
        def step(self, command, phase):
            super().step(command, phase)
            if level_before_exit and phase == 'level_released_bridge':
                point, rotation = ik.tcp_pose(command[:6], station.BASES['left'])
                predicted = measured.copy()
                predicted[:3, 3] += point + self.shift - initial_command[:3, 3]
                predicted[:3, :3] = rotation @ initial_command[:3, :3].T @ measured[:3, :3]
                self.frame_displacement = predicted @ np.linalg.inv(measured)
            self.samples.append((command.copy(), phase, self.frame_displacement.copy()))

    action = np.array(row['robot_action'])
    session = ExitSession(action, TRANSLATION_S_W, out, None)
    session.samples = []
    session.frame_displacement = np.eye(4)
    offset = 0 if side == 'left' else 7
    other = 'right' if side == 'left' else 'left'
    assert not preserve_held_aperture or side == 'left'
    assert not unload_during_opening or side == 'left'
    held_aperture = float(row['measured_joint_pos'][offset + 6])
    point, rotation = ik.tcp_pose(action[offset:offset + 6], station.BASES[side])
    measured = np.eye(4)
    position, orientation = ik.tcp_pose(np.array(row['measured_joint_pos'])[offset:offset + 6], station.BASES[side])
    measured[:3, 3], measured[:3, :3] = position + session.shift, orientation
    initial_command = np.eye(4)
    initial_command[:3, 3], initial_command[:3, :3] = point + session.shift, rotation
    if level_before_exit:
        assert side == 'left' and source_phase == 'withdraw_released_handle'
        bridge = transform(native['frame_body_poses'][names.index('Bridge')])
        target = level_bridge_target(bridge, measured, initial_command, 0.)
        session.move('left', target[:3, 3], target[:3, :3], .08, .75, 'level_released_bridge')
        session.hold(.25, 'hold_releveled_bridge')
        measured = session.frame_displacement @ measured
        point, rotation = target[:3, 3] - session.shift, target[:3, :3]
    if side == 'left':
        release_bridge(session, point + session.shift, rotation)
    else:
        release_handle(session, point + session.shift, rotation, measured)
    opening_goal = next(command[offset + 6] for command, phase, _ in reversed(session.samples)
                        if phase == ('open_release_bridge' if side == 'left' else 'open_supported_handle'))
    identity = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    moving_shapes = [prepare(shape, identity) for shape in robots[side]['colliders']]
    stationary = [prepare(shape, native[other + '_body_poses'][None, robots[other]['body_names'].index(shape['body'])])
                  for shape in robots[other]['colliders']]
    hardware = [prepare(shape, native['frame_body_poses'][None, names.index(shape['body'])])
                for shape in frame if shape['enabled']]
    released_hardware = hardware
    if released_handle:
        bridge = transform(native['frame_body_poses'][names.index('Bridge')])
        handle = handle_pose(bridge, np.deg2rad(-32.3), np.deg2rad(85.))
        connector = bridge.copy()
        connector[:3, 3] += bridge[:3, :3] @ [.030, 0., .052]
        connector[:3, :3] = bridge[:3, :3] @ Rotation.from_euler('x', -32.3, degrees=True).as_matrix()
        released_hardware = []
        for shape in hardware:
            if shape['body'] == 'Handle':
                shape = moved(shape, handle)
            elif shape['body'] == 'Connector':
                shape = moved(shape, connector)
            released_hardware.append(shape)
    table_points = np.array(list(itertools.product((-1., 1.), repeat=3))) * np.array(scene['table_size_m']) / 2
    table = prepare(dict(name='Table', body='Table', points=table_points),
                    np.array([[*scene['table_center_m'], 0., 0., 0., 1.]]))
    initial_geometry_joints = action[offset:offset + 7].copy()
    if preserve_held_aperture:
        initial_geometry_joints[6] = max(initial_geometry_joints[6], held_aperture)
    initial_links = link_transforms(initial_geometry_joints, side, session.shift)
    residuals = {}
    for body in robots[side]['body_names']:
        actual = transform(native[side + '_body_poses'][robots[side]['body_names'].index(body)])
        residuals[body] = np.linalg.inv(initial_links[body]) @ actual
    reports = []
    def scaled_deflections(fraction):
        deflections = {}
        for body, residual in residuals.items():
            delta = np.eye(4)
            delta[:3, 3] = fraction * residual[:3, 3]
            delta[:3, :3] = Rotation.from_rotvec(fraction * Rotation.from_matrix(residual[:3, :3]).as_rotvec()).as_matrix()
            deflections[body] = delta
        return deflections

    for fraction in (0., .5, 1.):
        deflections = scaled_deflections(fraction)
        hits, opening_contacts = [], []
        wing_gap, cloth_gap = float('inf'), float('inf')
        minimum_gap_after_slide = float('inf')
        for sample, (command, phase, displacement) in enumerate(session.samples):
            if unload_during_opening:
                if phase in ('level_released_bridge', 'hold_releveled_bridge'):
                    load_fraction = 1.
                elif phase == 'open_release_bridge':
                    progress = np.clip((command[6] - action[6]) / (opening_goal - action[6]), 0., 1.)
                    load_fraction = 1. + progress * (fraction - 1.)
                else:
                    load_fraction = fraction
                deflections = scaled_deflections(load_fraction)
            geometry_joints = command[offset:offset + 7].copy()
            if preserve_held_aperture and phase in ('level_released_bridge', 'hold_releveled_bridge',
                                                   'open_release_bridge'):
                # The crossbar physically stops the closed jaws. Preserve
                # its recorded aperture until the opening command exceeds it;
                # unloaded nominal closure would overlap the held object.
                geometry_joints[6] = max(geometry_joints[6], held_aperture)
            links = link_transforms(geometry_joints, side, session.shift)
            moving = [moved(shape, links[shape['body']] @ deflections[shape['body']]) for shape in moving_shapes]
            current_hardware = hardware if phase in ('level_released_bridge', 'hold_releveled_bridge',
                                'open_supported_handle', 'open_release_bridge') else released_hardware
            if level_before_exit:
                shifted_hardware = []
                for shape in current_hardware:
                    position, rotation = shape['data'][:2]
                    pose = np.eye(4)
                    pose[:3, 3], pose[:3, :3] = position[0], rotation[0]
                    shifted_hardware.append(moved(shape, displacement @ pose))
                current_hardware = shifted_hardware
            projected_frame = [projection_shape(shape) for shape in current_hardware]
            projected_wings = [shape for shape in projected_frame if shape['body'] in ('Left', 'Right')]
            projected_moving = [projection_shape(shape) for shape in moving]
            wing_gap = min(wing_gap, shape_clearance(projected_wings, projected_moving, .002))
            cloth_gap = min(cloth_gap, cloth_clearance(native['positions'][None], faces, projected_moving))
            hand = [shape for shape in projected_moving if shape['body'] in ('gripper', 'tip_left', 'tip_right')]
            hand_gap = min(shape_clearance(projected_frame, hand),
                           cloth_clearance(native['positions'][None], faces, hand))
            if phase in ('withdraw_release_support', 'withdraw_released_handle'):
                minimum_gap_after_slide = min(minimum_gap_after_slide, hand_gap)
            second = current_hardware + stationary + [table]
            low_a, high_a = (np.array([shape[key] for shape in moving]) for key in ('global_low', 'global_high'))
            low_b, high_b = (np.array([shape[key] for shape in second]) for key in ('global_low', 'global_high'))
            overlap = np.all(high_a[:, None] >= low_b, axis=2) & np.all(high_b >= low_a[:, None], axis=2)
            for a, b in np.argwhere(overlap):
                one, two = moving[a], second[b]
                if one['body'] == 'base' and two['body'] == 'Table':
                    continue
                depth, _, _ = compare(one, two)
                if depth <= .001:
                    continue
                hit = dict(sample=sample, phase=phase, first=one['name'],
                           second=two['body'] + '/' + two['name'], depth_bound_m=depth)
                opening_phase = 'open_release_bridge' if side == 'left' else 'open_supported_handle'
                held_body = 'Bridge' if side == 'left' else 'Handle'
                intentional = phase == opening_phase or (level_before_exit and phase in (
                    'level_released_bridge', 'hold_releveled_bridge'))
                if intentional and one['body'] in ('tip_left', 'tip_right') and two['body'] == held_body:
                    opening_contacts.append(hit)
                else:
                    hits.append(hit)
        reports.append(dict(recorded_deflection_fraction=fraction, hits=hits,
                            intentional_opening_contacts=opening_contacts,
                            minimum_wing_gap_m=wing_gap, minimum_cloth_gap_m=cloth_gap,
                            minimum_hand_gap_after_slide_m=minimum_gap_after_slide,
                            final_hand_gap_m=hand_gap,
                            passed=not hits and wing_gap >= .002 and cloth_gap >= .002 and hand_gap >= .02))
    report = dict(passed=all(item['passed'] for item in reports), models=reports,
                  source_run=run.name, source_control_step=index, source_time_s=row['time_s'], source_phase=source_phase,
                  control_steps=session.steps, moving_arm=side, released_handle_model=released_handle,
                  level_before_exit=level_before_exit,
                  preserve_held_aperture=preserve_held_aperture,
                  unload_during_opening=unload_during_opening,
                  deflection_interpretation=('Full recorded link residual during the held correction; linearly relaxes with the first jaw-opening command to each reported residual fraction, then stays there.'
                                            if unload_during_opening else 'Each reported residual fraction is constant over the complete planned motion.'),
                  recorded_held_aperture=held_aperture if preserve_held_aperture else None,
                  scope='Planning only. Cloth and the opposite arm stay at recorded native poses. Without leveling, the frame stays fixed too. With leveling, the frame follows a rigid measured left grasp during the correction and then stays at that predicted pose. The moving arm follows ordinary commands with zero, half or all of its recorded per-link residual. Intentional grasp/opening contact is disclosed separately. The optional released-handle model uses gimbal angles[-32.3,85]degrees after opening. Dynamic deflection, friction and cloth motion are not certified.',
                  input_sha256={str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (run / 'steps.jsonl', native_file, run / 'robot-cooked-colliders.json',
                                             run / 'frame-geometry.json', run / 'geometry.npz')},
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('preflight_bridge_exit.py', 'supported_release.py', 'robot_session.py')})
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'models'}))
    for model in reports:
        print(json.dumps({key: value for key, value in model.items() if key not in ('hits', 'intentional_opening_contacts')}))
        print('potential_collisions', len(model['hits']), model['hits'][:4])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source-phase', choices=('hold_release_support', 'hold_releveled_bridge',
                        'withdraw_released_handle', 'withdraw_release_support'), default='hold_release_support')
    parser.add_argument('--side', choices=('left', 'right'), default='left')
    parser.add_argument('--released-handle', action='store_true')
    parser.add_argument('--level-before-exit', action='store_true')
    parser.add_argument('--preserve-held-aperture', action='store_true')
    parser.add_argument('--unload-during-opening', action='store_true')
    args = parser.parse_args()
    root = OUTPUT_ROOT
    assert args.run.resolve().parent == args.out.resolve().parent == root
    args.out.mkdir(exist_ok=False)
    preflight(args.run, args.out, args.source_phase, args.side, args.released_handle,
              args.level_before_exit, args.preserve_held_aperture, args.unload_during_opening)
