"""Place the installed mop with its handle upright using ordinary wrist targets."""

import json

import numpy as np
from scipy.spatial.transform import Rotation


def upright_target(bridge, handle, wrist):
    """Preserve the measured contact grasp while aiming the shaft vertically."""
    from robot_reopen import handle_pose

    axis = bridge[:3, :3].T @ np.array([0., 0., 1.])
    x = np.arctan2(-axis[1], axis[2])
    y = np.arcsin(np.clip(axis[0], -1., 1.))
    assert max(abs(x), abs(y)) < np.deg2rad(80.), 'Head tilt leaves insufficient gimbal margin'
    return handle_pose(bridge, x, y) @ np.linalg.inv(handle) @ wrist


def inclination(handle):
    """Measure the shaft angle from world vertical in degrees."""
    return float(np.rad2deg(np.arccos(np.clip(handle[2, 2], -1., 1.))))


def flat_placement_target(bridge, handle, wrist, table_top):
    """Aim the measured grasp at an upright handle above a level, supported head."""
    from robot_reopen import handle_pose

    goal = bridge.copy()
    yaw = Rotation.from_matrix(bridge[:3, :3]).as_euler('xyz')[2]
    goal[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    goal[2, 3] = table_top + .005
    return handle_pose(goal, 0., 0.) @ np.linalg.inv(handle) @ wrist


def place_upright(session, bridge, handle, wrist, table_top, read_frame=None, measure_wrist=None):
    """Orient, lower, correct and release the handle without object-state writes."""
    from wrist_tracking import limited

    target = upright_target(bridge, handle, wrist)
    session.move('right', target[:3, 3], target[:3, :3], .08, 2., 'upright_handle_for_release')
    session.hold(.5, 'hold_upright_handle')
    # This table location gives the low upright grasp adequate wrist margin.
    # It changes only final placement, after the completed return/hold motion.
    place = target[:3, 3] + [.10, .10, table_top + .005 - bridge[2, 3]]
    rotation = target[:3, :3].copy()
    session.move('right', place, rotation, .08, 3., 'set_down_frame')
    records = []
    for index in range(6):
        current = None if read_frame is None else read_frame()
        angle = None if current is None else inclination(current['Handle'])
        head_angle = None if current is None else inclination(current['Bridge'])
        height = None if current is None else float(current['Bridge'][2, 3] - table_top)
        if (angle is not None and angle <= 1. and head_angle <= 2.
                and -.001 <= height <= .007):
            break
        if current is not None:
            measured = measure_wrist('right')
            desired = flat_placement_target(current['Bridge'], current['Handle'], measured, table_top)
            delta = limited(.8 * (desired[:3, 3] - measured[:3, 3]), .010)
            angular = Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).as_rotvec()
            place += delta
            rotation = Rotation.from_rotvec(limited(.8 * angular, np.deg2rad(3.))).as_matrix() @ rotation
        session.move('right', place, rotation, .08, .5, 'level_handle_before_release')
        records.append(dict(iteration=index, inclination_before_deg=angle,
            head_inclination_before_deg=head_angle, bridge_height_before_m=height,
            inclination_after_deg=None if read_frame is None else inclination(read_frame()['Handle'])))
    final = None if read_frame is None else read_frame()
    final_angle = None if final is None else inclination(final['Handle'])
    final_head_angle = None if final is None else inclination(final['Bridge'])
    final_height = None if final is None else float(final['Bridge'][2, 3] - table_top)
    (session.out / 'upright-release.json').write_text(json.dumps(dict(
        ordinary_wrist_commands_only=True, final_placement_offset_xy_m=[.10, .10],
        desired_handle_inclination_deg=0., maximum_corrections=6,
        correction_translation_limit_m=.010, correction_rotation_limit_deg=3.,
        final_handle_inclination_deg=final_angle, corrections=records,
        final_head_inclination_deg=final_head_angle, final_bridge_height_m=final_height,
        maximum_duration_s=17.5), indent=2))
    assert (final is None or (final_angle <= 2. and final_head_angle <= 3.
            and -.001 <= final_height <= .008)), 'Assembly remains too tilted or elevated for release'
    session.move('right', place, rotation, .6, 1., 'release_handle')
    session.move('right', place - .04 * rotation[:, 2] + [0., 0., .02], rotation, .6, .75, 'clear_upright_shaft')
    from robot_reopen import free_retreat_rotation
    retreat = place + [-.10, -.12, .15]
    free_rotation = free_retreat_rotation(session, retreat, 'right')
    session.move('right', retreat, free_rotation, .6, 1.25, 'withdraw')
    session.hold(6., 'released')
