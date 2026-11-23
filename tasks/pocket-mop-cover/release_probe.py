"""Initialize a short contact diagnostic from a recorded held assembly.

This is a cold solver start with zero velocities, not an episode replay. All
state writes occur before the action recorder starts. These runs cannot score.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_reopen import transform


def check_configuration(original, current):
    """Allow only explicit numerical refinement while preserving physical parameters."""
    refinable = ('physics_hz', 'iterations', 'contact_updates', 'contact_subiterations')
    changes = {}
    for key in (*refinable, 'spacing', 'ccd', 'product_profile', 'table_origin', 'snap_throat',
                'robot_cover', 'cover_offset_x', 'mouth_height'):
        before, after = original.get(key), current.get(key)
        if current.get('release_probe_refine_contact') and key in refinable:
            assert after >= before, f'Diagnostic refinement cannot reduce {key}'
        else:
            assert before == after, f'Diagnostic configuration differs: {key}'
        if before != after:
            changes[key] = dict(original=before, diagnostic=after)
    assert not current.get('release_probe_refine_contact') or changes, 'Refinement needs an increased numerical setting'
    return changes


def finger_joints_from_links(poses, names):
    """Recover each prismatic finger position independently from its native pose."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN
    from yam_arena.urdf import forward_kinematics

    closed = dict.fromkeys((*ARM_JOINTS, *FINGER_JOINTS), 0.)
    opened = dict(closed, **dict.fromkeys(FINGER_JOINTS, FINGER_OPEN))
    zero = forward_kinematics(YAM_URDF, closed)
    one = forward_kinematics(YAM_URDF, opened)
    actual_parent = transform(poses[names.index('gripper')])
    result = {}
    for joint, body in zip(FINGER_JOINTS, ('tip_left', 'tip_right'), strict=True):
        initial = np.linalg.inv(zero['gripper']) @ zero[body]
        full = np.linalg.inv(one['gripper']) @ one[body]
        measured = np.linalg.inv(actual_parent) @ transform(poses[names.index(body)])
        axis = full[:3, 3] - initial[:3, 3]
        opening = float((measured[:3, 3] - initial[:3, 3]) @ axis / (axis @ axis))
        residual = measured[:3, 3] - initial[:3, 3] - opening * axis
        assert np.linalg.norm(residual) < .0001, 'Finger does not match the stock prismatic geometry'
        angle = Rotation.from_matrix(measured[:3, :3] @ initial[:3, :3].T).magnitude()
        assert angle < .001, 'Finger orientation differs from its stock joint'
        assert -.001 <= opening <= 1.001, 'Recorded finger is outside its physical range'
        result[joint] = opening * FINGER_OPEN
    return result


def initialize(base, frame, cover, request, out):
    """Write the diagnostic start once, before any recorded control or physics step."""
    import torch
    from yam_arena.constants import ARM_JOINTS

    source = out.parent / request['release_probe_run']
    without_cover = bool(request.get('release_probe_without_cover'))
    assert (cover is None) == without_cover
    assert source.resolve().parent == out.parent.resolve()
    assert json.loads((source / 'result.json').read_text())['status'] == 'completed'
    original = json.loads((source / 'request.json').read_text())
    assert not original.get('release_probe_run'), 'Probe starts must come from full recordings'
    refinements = check_configuration(original, request)
    for name in ('versions.json', 'frame-geometry.json', 'scene-audit.json'):
        before = json.loads((source / name).read_text())
        current = json.loads((out / name).read_text())
        before.pop('diagnostic_initialization', None)
        current.pop('diagnostic_initialization', None)
        if name == 'scene-audit.json' and without_cover:
            assert before['cloth'] and not current['cloth']
            assert current['cover_center_m'] is None
            before['cloth'] = False
            before['cover_center_m'] = None
        assert before == current, f'Diagnostic scene differs: {name}'
    original_hashes = json.loads((source / 'manifest.json').read_text())['files']
    for name in ('frame_native.py', 'frame_asset.py', 'native.py', 'mesh.py', 'cover_mesh.py',
                 'parameters.json', 'station_coordinates.py'):
        assert original_hashes[name] == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (source / 'steps.jsonl').read_text().splitlines()]
    row = next(row for row in reversed(rows) if row['phase'] == 'hold_releveled_bridge')
    source_hz = original['physics_hz']
    index = (row['step'] + 1) * (source_hz // 20) - 1
    path = source / 'physics-chunks' / f'{index // source_hz:05d}.npz'
    with np.load(path) as chunk:
        state = {name: chunk[name][index % source_hz].copy() for name in
                 ('positions', 'frame_body_poses', 'frame_joint_positions', 'left_body_poses', 'right_body_poses')}
    robot_row = row
    robot_path = path
    if request.get('release_probe_hands_clear'):
        robot_row = rows[-1]
        robot_index = (robot_row['step'] + 1) * (source_hz // 20) - 1
        robot_path = source / 'physics-chunks' / f'{robot_index // source_hz:05d}.npz'
        with np.load(robot_path) as chunk:
            for name in ('left_body_poses', 'right_body_poses'):
                state[name] = chunk[name][robot_index % source_hz].copy()
    frame_names = json.loads((source / 'frame-names.json').read_text())
    assert frame_names['body_names'] == list(frame.body_names)
    assert frame_names['joint_names'] == list(frame.joint_names)
    robots = json.loads((source / 'robot-cooked-colliders.json').read_text())
    native_fingers = {}

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=base.device)

    bridge = state['frame_body_poses'][frame.body_names.index('Bridge')]
    # This installed Isaac Lab backend uses xyzw, matching the native recording.
    root = bridge[None]
    frame.write_root_link_pose_to_sim_index(root_pose=tensor(root))
    frame.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1, 6))))
    q = tensor(state['frame_joint_positions'][None])
    frame.write_joint_position_to_sim_index(position=q)
    frame.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    for side, offset in (('left', 0), ('right', 7)):
        robot = base.scene[side + '_robot']
        positions = dict(zip(ARM_JOINTS, robot_row['measured_joint_pos'][offset:offset + 6], strict=True))
        fingers = finger_joints_from_links(state[side + '_body_poses'], robots[side]['body_names'])
        positions.update(fingers)
        native_fingers[side] = fingers
        assert set(positions) == set(robot.joint_names)
        q = tensor([[positions[name] for name in robot.joint_names]])
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    if cover is not None:
        nodes = np.concatenate((state['positions'], np.zeros_like(state['positions'])), axis=1)[None]
        cover.obj.write_nodal_state_to_sim_index(tensor(nodes))
        cover.obj.reset()
        cover.bind(base)
    # Reset normally performs this forward pass. Our diagnostic writes happen
    # after that reset, so synchronize child-link poses without a physics step.
    base.sim.forward()
    from robot_gate import array
    restored_root = array(frame.root_view.get_root_transforms())[0]
    root_error = float(np.linalg.norm(restored_root[:3] - bridge[:3]))
    root_angle = float((Rotation.from_quat(restored_root[3:]) *
                        Rotation.from_quat(bridge[3:]).inv()).magnitude())
    node_error = None
    if cover is not None:
        restored_nodes, restored_velocities = cover.nodes()
        node_error = float(abs(restored_nodes - state['positions']).max())
        assert node_error < 1e-6 and not np.any(restored_velocities), 'Native cloth initialization differs'
    link_errors = {}
    for name, rig, key in [('frame', frame, 'frame_body_poses'),
                           ('left', base.scene['left_robot'], 'left_body_poses'),
                           ('right', base.scene['right_robot'], 'right_body_poses')]:
        restored = array(rig.root_view.get_link_transforms())[0].reshape(-1, 7)
        expected = state[key]
        distance = np.linalg.norm(restored[:, :3] - expected[:, :3], axis=1)
        angles = (Rotation.from_quat(restored[:, 3:]) * Rotation.from_quat(expected[:, 3:]).inv()).magnitude()
        link_errors[name] = dict(max_translation_m=float(distance.max()), max_rotation_rad=float(angles.max()))
    (out / 'release-probe-link-readback.json').write_text(json.dumps(link_errors, indent=2))
    assert root_error < 1e-6 and root_angle < 1e-5, 'Native root readback differs from the recorded xyzw pose'
    assert all(item['max_translation_m'] < .0002 and item['max_rotation_rad'] < .002
               for item in link_errors.values()), 'Restored articulated links differ from the recorded held pose'
    # The audit's first finite difference must start at this diagnostic state;
    # keep the original material rest shape and faces untouched.
    if cover is not None:
        np.savez_compressed(out / 'geometry.npz', rest=cover.rest, initial=state['positions'], faces=cover.faces)
    proof = dict(source_run=source.name, source_control_step=row['step'], source_time_s=row['time_s'],
                 source_phase=row['phase'], cold_solver_start=True, all_initial_velocities_zero=True,
                 cover_omitted_diagnostic_control=without_cover,
                 numerical_refinements=refinements,
                 robot_source_control_step=robot_row['step'],
                 robot_source_time_s=robot_row['time_s'],
                 hands_initialized_clear=bool(request.get('release_probe_hands_clear')),
                 native_readback_errors=dict(root_translation_m=root_error,
                     root_rotation_rad=root_angle, cloth_position_m=node_error, links=link_errors),
                 finger_joints_reconstructed_individually=native_fingers,
                 writes_only_before_action_recording=True, benchmark_eligible=False,
                 limitations='Contact warm-start impulses are not restored. This isolates a release motion and cannot demonstrate the full task or independent episode replay.',
                 input_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                               for name in ('steps.jsonl', 'frame-names.json', 'robot-cooked-colliders.json',
                                            str(path.relative_to(source)), str(robot_path.relative_to(source)), 'manifest.json')})
    (out / 'release-probe-initialization.json').write_text(json.dumps(proof, indent=2))
    initial = dict(row, robot_action=robot_row['robot_action'], measured_joint_pos=robot_row['measured_joint_pos'])
    return initial, rows[row['step'] + 1:]


def execute(session, initial, tail, read_frame):
    """Warm the ordinary held command, then test a baseline or revised release."""
    from supported_release import handle_clear_of_left_exit, release_bridge
    from upright_release import inclination
    from yam_arena import ik, station

    session.action = np.asarray(initial['robot_action']).copy()
    session.hold(1., 'settle_release_probe')
    current = read_frame()
    before = dict(head_inclination_deg=inclination(current['Bridge']),
                  head_height_m=float(current['Bridge'][2, 3]))
    report = dict(diagnostic_only=True, benchmark_eligible=False, after_warmup=before,
                  cover_omitted_diagnostic_control=bool(session.request.get('release_probe_without_cover')),
                  hands_initialized_clear=bool(session.request.get('release_probe_hands_clear')),
                  baseline_saved_actions=session.request['release_probe_baseline'])
    (session.out / 'release-probe.json').write_text(json.dumps(report, indent=2))
    if session.request.get('release_probe_hands_clear'):
        session.hold(6.15, 'released')
    elif session.request['release_probe_baseline']:
        assert before['head_inclination_deg'] <= 5. and -.001 <= before['head_height_m'] <= .010
        assert handle_clear_of_left_exit(current['Bridge'], current['Handle'])
        for row in tail:
            session.step(np.asarray(row['robot_action']), row['phase'])
    else:
        assert before['head_inclination_deg'] <= 5. and -.001 <= before['head_height_m'] <= .010
        assert handle_clear_of_left_exit(current['Bridge'], current['Handle'])
        point, rotation = ik.tcp_pose(session.action[:6], station.BASES['left'])
        release_bridge(session, point + session.shift, rotation)
        session.hold(4., 'released')
    current = read_frame()
    report['after_release'] = dict(head_inclination_deg=inclination(current['Bridge']),
                                   head_height_m=float(current['Bridge'][2, 3]))
    (session.out / 'release-probe.json').write_text(json.dumps(report, indent=2))
