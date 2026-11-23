"""Set down the mop, grasp its bridge, and clear the lowered handle."""

import json

import numpy as np
from scipy.spatial.transform import Rotation


PLACEMENT_OFFSET = np.zeros(3)
"""Keep final placement vertical; lateral motion after cloth contact can unseat a pocket."""


def support_target(bridge):
    """Aim a downward pinch at the exposed crossbar and upright opposite the right hand."""
    point = bridge[:3, 3] + bridge[:3, :3] @ np.array([.030, .043, .030])
    return point, bridge[:3, :3] @ np.diag([-1., 1., -1.])


def handle_angles(bridge, handle):
    """Read the passive gimbal angles from the measured body orientations."""
    axis = bridge[:3, :3].T @ handle[:3, 2]
    return np.array([np.arctan2(-axis[1], axis[2]),
                     np.arcsin(np.clip(axis[0], -1., 1.))])


def ready_to_release(bridge, handle, table_top):
    """Require a supported head and a handle lowered along its long axis."""
    from upright_release import inclination

    angles = np.rad2deg(handle_angles(bridge, handle))
    return bool(inclination(bridge) <= 5. and -.001 <= bridge[2, 3] - table_top <= .010
                and abs(angles[0]) <= 12. and 78. <= angles[1] <= 85.1)


def handle_clear_of_left_exit(bridge, handle):
    """Require the free handle to lean away from the positive-Y support-hand exit."""
    x, y = np.rad2deg(handle_angles(bridge, handle))
    return bool(-1e-8 <= x <= 85.1 and 78. <= y <= 85.1)


def level_bridge_target(bridge, measured, command, table_top):
    """Correct a held head to the table while preserving the measured contact grasp."""
    goal = bridge.copy()
    yaw = Rotation.from_matrix(bridge[:3, :3]).as_euler('xyz')[2]
    goal[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    goal[2, 3] = table_top + .005
    desired = goal @ np.linalg.inv(bridge) @ measured
    displacement = desired[:3, 3] - measured[:3, 3]
    rotation = desired[:3, :3] @ measured[:3, :3].T
    assert np.linalg.norm(displacement) <= .020, 'Release leveling exceeds the bounded wrist displacement'
    assert Rotation.from_matrix(rotation).magnitude() <= np.deg2rad(15.), 'Release leveling exceeds the bounded wrist rotation'
    target = command.copy()
    target[:3, 3] += displacement
    target[:3, :3] = rotation @ command[:3, :3]
    return target


def open_hand_stationary(session, side, opening, seconds, phase):
    """Open the jaws while preserving every arm joint target exactly."""
    from yam_arena import ik

    index = 6 if side == 'left' else 13
    initial = session.action.copy()
    count = round(seconds * 20)
    for step in range(1, count + 1):
        command = initial.copy()
        command[index] += ik.smoothstep(step / count) * (opening - initial[index])
        session.step(command, phase)


def release_bridge(session, point, rotation):
    """Open, center the wide jaws over the rail, then raise the cleared hand."""
    # The crossbar grasp is at local X=30 mm, but RailR is centered at10.5 mm.
    # This screened path clears the bare frame. The covered diagnostic still
    # tips: the rail rises from below the fingertip as textile contact unloads.
    # Keep the same ordinary path while diagnosing that material interaction.
    center = rotation[:, 0].copy()
    center[2] = 0.
    assert center[0] < 0., 'Rail centering must move toward the negative-X side'
    center *= .020 / np.linalg.norm(center)
    open_hand_stationary(session, 'left', 1., .5, 'open_release_bridge')
    session.move('left', point + center, rotation, 1., .3, 'center_release_gripper')
    session.move('left', point + center + [0., 0., .08], rotation, 1., .75, 'clear_release_support')
    session.move('left', point + center + [0., .11, .10], rotation, 1., .6, 'withdraw_release_support')


def release_handle(session, point, rotation, measured=None):
    """Open while backing away from the support hand using the measured wrist pose."""
    from robot_reopen import free_retreat_rotation

    if measured is not None:
        point, rotation = measured[:3, 3], measured[:3, :3]
    back = -rotation[:, 2]
    session.move('right', point + .01 * back, rotation, .5, .3, 'open_supported_handle')
    # R118's biased wrist rose 7.5 mm while opening; its horizontal escape
    # subsequently contacted the right frame end. The measured pose removes
    # that preload, and reversing the full approach axis also clears upward.
    away = .07 * back
    session.move('right', point + away, rotation, .7, .65, 'clear_supported_handle')
    retreat = point + [-.10, -.12, .15]
    session.move('right', retreat, free_retreat_rotation(session, retreat, 'right'),
                 .7, 1., 'withdraw_released_handle')


def place_supported(session, place, rotation, bridge, handle, wrist, table_top,
                    read_frame=None, measure_wrist=None):
    """Use ordinary robot contact to place and release the installed cover."""
    from robot_reopen import handle_pose
    from upright_release import inclination
    from wrist_tracking import limited
    from yam_arena import ik, station

    def commanded_wrist(side='right'):
        offset = 0 if side == 'left' else 7
        point, orient = ik.tcp_pose(session.action[offset:offset + 6], station.BASES[side])
        result = np.eye(4)
        result[:3, 3], result[:3, :3] = point + session.shift, orient
        return result

    placement_offset = PLACEMENT_OFFSET.copy()
    session.move('right', place + placement_offset, rotation, .08, 3., 'set_down_frame')
    if read_frame is None:
        bridge = bridge.copy()
        bridge[:3, 3] += placement_offset
        bridge[2, 3] = table_top + .005
        bridge[:3, :3] = Rotation.from_euler('z', Rotation.from_matrix(
            bridge[:3, :3]).as_euler('xyz')[2]).as_matrix()
    else:
        bridge = read_frame()['Bridge']
    point, downward = support_target(bridge)
    session.move('left', point + [0., 0., .08], downward, .35, 1.5,
                 'approach_release_support', joint=True)
    session.move('left', point, downward, .35, .75, 'enter_release_bridge')
    session.move('left', point, downward, .08, .5, 'grasp_release_bridge')
    if read_frame is not None:
        bridge = read_frame()['Bridge']
    goal = bridge.copy()
    goal[:3, :3] = Rotation.from_euler('z', Rotation.from_matrix(
        bridge[:3, :3]).as_euler('xyz')[2]).as_matrix()
    goal[2, 3] = table_top + .005
    targets = {}
    bridge_grasps = {}
    for side in ('left', 'right'):
        command = commanded_wrist(side)
        measured = command if measure_wrist is None else measure_wrist(side)
        local = np.linalg.inv(bridge) @ measured
        desired = goal @ local
        bridge_grasps[side] = local.tolist()
        command[:3, 3] += desired[:3, 3] - measured[:3, 3]
        command[:3, :3] = desired[:3, :3] @ measured[:3, :3].T @ command[:3, :3]
        targets[side] = (command[:3, 3], command[:3, :3], .08)
    session.move_both(targets, .75, 'level_grasped_bridge')
    point, downward = targets['left'][:2]

    local_grasp = np.linalg.inv(handle) @ wrist
    current = None if read_frame is None else read_frame()
    if current is not None:
        bridge, handle = current['Bridge'], current['Handle']
        wrist = measure_wrist('right')
        local_grasp = np.linalg.inv(handle) @ wrist
    desired = handle_pose(bridge, 0., np.deg2rad(82.)) @ local_grasp
    command = commanded_wrist()
    if current is None:
        command = desired.copy()
    else:
        command[:3, 3] += desired[:3, 3] - wrist[:3, 3]
        command[:3, :3] = desired[:3, :3] @ wrist[:3, :3].T @ command[:3, :3]
    session.move('right', command[:3, 3], command[:3, :3], .08, 2., 'lower_supported_handle')
    corrections = []
    for index in range(3):
        current = None if read_frame is None else read_frame()
        angles = None if current is None else np.rad2deg(handle_angles(current['Bridge'], current['Handle']))
        if angles is not None and abs(angles[0]) <= 4. and 80. <= angles[1] <= 84.:
            break
        if current is not None:
            measured = measure_wrist('right')
            grasp = np.linalg.inv(current['Handle']) @ measured
            desired = handle_pose(current['Bridge'], 0., np.deg2rad(82.)) @ grasp
            command[:3, 3] += limited(.8 * (desired[:3, 3] - measured[:3, 3]), .006)
            angular = Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).as_rotvec()
            command[:3, :3] = Rotation.from_rotvec(limited(.8 * angular, np.deg2rad(3.))).as_matrix() @ command[:3, :3]
        session.move('right', command[:3, 3], command[:3, :3], .08, .4, 'align_supported_handle')
        corrections.append(dict(iteration=index, angles_before_deg=None if angles is None else angles.tolist()))
    session.hold(.25, 'hold_release_support')
    final = None if read_frame is None else read_frame()
    head_angle = None if final is None else inclination(final['Bridge'])
    height = None if final is None else float(final['Bridge'][2, 3] - table_top)
    angles = None if final is None else np.rad2deg(handle_angles(final['Bridge'], final['Handle']))
    release_report = dict(
        ordinary_wrist_commands_only=True,
        final_placement_offset_m=placement_offset.tolist(),
        physical_support='Ordinary parallel-jaw pinch on existing Bridge/Crossbar and UprightR',
        desired_handle_gimbal_deg=[0., 82.], head_angle_before_release_deg=head_angle,
        head_height_before_release_m=height, gimbal_angles_before_release_deg=None if angles is None else angles.tolist(),
        measured_bridge_relative_wrist_at_grasp=bridge_grasps, handle_corrections=corrections,
        release_order='Right handle hand clears while left crossbar grasp supports the head; then left releases',
        maximum_duration_s=19.05)
    (session.out / 'supported-release.json').write_text(json.dumps(release_report, indent=2))
    assert (final is None or ready_to_release(final['Bridge'], final['Handle'], table_top)), 'Supported assembly is not ready for release'
    released_right = None if measure_wrist is None else measure_wrist('right')
    if released_right is not None:
        release_report['right_release'] = dict(
            measured_wrist_pose=released_right.tolist(),
            removed_command_translation_bias_m=(command[:3, 3] - released_right[:3, 3]).tolist(),
            opening_retreat_m=(-.01 * released_right[:3, 2]).tolist(),
            approach_axis_retreat_m=(-.07 * released_right[:3, 2]).tolist())
        (session.out / 'supported-release.json').write_text(json.dumps(release_report, indent=2))
    release_handle(session, command[:3, 3], command[:3, :3], released_right)
    after_right = None if read_frame is None else read_frame()
    if after_right is not None:
        angle = inclination(after_right['Bridge'])
        height = float(after_right['Bridge'][2, 3] - table_top)
        release_report['after_right_withdrawal'] = dict(
            head_inclination_deg=angle, head_height_m=height,
            handle_gimbal_deg=np.rad2deg(handle_angles(after_right['Bridge'], after_right['Handle'])).tolist())
        (session.out / 'supported-release.json').write_text(json.dumps(release_report, indent=2))
        assert angle <= 15. and -.001 <= height <= .020, 'Head moved beyond the bounded leveling range'
    # R120 retained the head but it rolled 7.2 degrees within the pinch when
    # the right hand unloaded. Correct through the remaining contact grasp;
    # keep the original release-posture guard after this physical motion.
    current_bridge = goal if after_right is None else after_right['Bridge']
    command = commanded_wrist('left')
    measured = command if measure_wrist is None else measure_wrist('left')
    leveled = level_bridge_target(current_bridge, measured, command, table_top)
    session.move('left', leveled[:3, 3], leveled[:3, :3], .08, .75, 'level_released_bridge')
    session.hold(.25, 'hold_releveled_bridge')
    release_report['left_relevel'] = dict(
        command_translation_correction_m=(leveled[:3, 3] - command[:3, 3]).tolist(),
        command_rotation_correction_deg=float(np.rad2deg(Rotation.from_matrix(
            leveled[:3, :3] @ command[:3, :3].T).magnitude())),
        duration_s=1., maximum_translation_m=.020, maximum_rotation_deg=15.)
    after_level = None if read_frame is None else read_frame()
    if after_level is not None:
        angle = inclination(after_level['Bridge'])
        height = float(after_level['Bridge'][2, 3] - table_top)
        handle_clear = handle_clear_of_left_exit(after_level['Bridge'], after_level['Handle'])
        release_report['after_left_relevel'] = dict(
            head_inclination_deg=angle, head_height_m=height,
            handle_gimbal_deg=np.rad2deg(handle_angles(after_level['Bridge'], after_level['Handle'])).tolist(),
            handle_clear_of_left_exit=handle_clear)
        (session.out / 'supported-release.json').write_text(json.dumps(release_report, indent=2))
        assert angle <= 5. and -.001 <= height <= .010, 'Head is not level after supported release correction'
        assert handle_clear, 'Free handle obstructs the screened left-hand exit'
    released_wrist = None if measure_wrist is None else measure_wrist('left')
    if released_wrist is not None:
        release_report['left_release'] = dict(
            measured_wrist_pose=released_wrist.tolist(),
            retained_command_translation_bias_m=(leveled[:3, 3] - released_wrist[:3, 3]).tolist(),
            stationary_arm_joint_targets_during_opening=True)
        (session.out / 'supported-release.json').write_text(json.dumps(release_report, indent=2))
    release_bridge(session, leveled[:3, 3], leveled[:3, :3])
    after_left = None if read_frame is None else read_frame()
    if after_left is not None:
        release_report['after_left_withdrawal'] = dict(
            head_inclination_deg=inclination(after_left['Bridge']),
            head_height_m=float(after_left['Bridge'][2, 3] - table_top),
            handle_gimbal_deg=np.rad2deg(handle_angles(after_left['Bridge'], after_left['Handle'])).tolist())
        (session.out / 'supported-release.json').write_text(json.dumps(release_report, indent=2))
    # The leveling second comes from the final dwell, retaining the existing
    # worst-case action bound and the scorer's unchanged two-second hold.
    session.hold(4., 'released')
