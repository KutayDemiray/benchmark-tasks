"""CPU kinematic clearance bounds for planning around the physical handle."""

import numpy as np


def link_transforms(joints, side, shift):
    """Return stock URDF link poses for one measured or commanded 7D arm state."""
    from yam_arena import YAM_URDF, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN
    from yam_arena.urdf import forward_kinematics
    values = dict(zip(ARM_JOINTS, joints[:6], strict=True))
    values.update({name: float(joints[6] * FINGER_OPEN) for name in FINGER_JOINTS})
    poses = forward_kinematics(YAM_URDF, values)
    for pose in poses.values():
        pose[:3, 3] += np.asarray(station.BASES[side]) + shift
    return poses


def prepare_convexes(geometry):
    """Cache exact projections of each authored convex hull on its normals."""
    prepared = []
    for shape in geometry['colliders']:
        normals = np.asarray(shape['planes'])[:, :3]
        projection = np.asarray(shape['points']) @ normals.T
        prepared.append(dict(name=shape['name'], body=shape['body'], normals=normals,
                             lower=projection.min(0), upper=projection.max(0)))
    return prepared


def capsule_contacts(poses, shapes, center, axis, radius, height, margin=0.):
    """Return conservative capsule-overlap bounds against kinematic link hulls.

    A zero bound certifies separation along a hull normal. Positive values are
    potential contacts, not a replacement for native all-step contact audits.
    A positive margin inflates the planning capsule without modifying physics.
    """
    contacts = []
    for shape in shapes:
        pose = poses[shape['body']]
        local_center = (center-pose[:3, 3]) @ pose[:3, :3]
        local_axis = axis @ pose[:3, :3]
        middle = shape['normals'] @ local_center
        extent = height/2 * abs(shape['normals'] @ local_axis) + radius + margin
        bound = np.minimum(middle+extent-shape['lower'], shape['upper']-(middle-extent)).min()
        if bound > 0:
            contacts.append(dict(name=shape['name'], body=shape['body'], depth_upper_bound_m=float(bound)))
    return contacts


def certify_handle_approach(initial, motions, rotation, geometry, center, axis, shift):
    """Check the proposed ordinary approach against a handle inflated by 10 mm."""
    from robot_gate import solve_pose
    from yam_arena import ik, station
    shapes = prepare_convexes(geometry)
    state = np.asarray(initial).copy()
    checked = 0
    for goal, seconds, joint in motions:
        goal = np.asarray(goal)
        q0 = state[:6].copy()
        q = q0.copy()
        p0, r0 = ik.tcp_pose(q0, station.BASES['right'])
        p0 += shift
        qgoal, _, _ = solve_pose(goal-shift, rotation, q0, 'right')
        g0 = state[6]
        count = round(seconds*20)
        for index in range(1, count+1):
            s = ik.smoothstep(index/count)
            if joint:
                q = q0+s*(qgoal-q0)
            else:
                point = p0+s*(goal-p0)
                interpolated = ik.slerp_matrix(r0, rotation, s)
                q, pe, re = ik.solve(point-shift, ik.matrix_to_quat_xyzw(interpolated), q,
                                    base_pos=station.BASES['right'], iters=120)
                if pe >= .004 or re >= .04:
                    q, _, _ = solve_pose(point-shift, interpolated, q, 'right')
            state = np.r_[q, g0+s*(.6-g0)]
            contacts = capsule_contacts(link_transforms(state, 'right', shift), shapes,
                                        center, axis, .011, .478, margin=.01)
            assert not contacts, f'Proposed approach may contact handle: {contacts[:2]}'
            checked += 1
    return dict(passed=True, control_samples=checked, planning_margin_m=.01,
                colliders=len(shapes), scope='Kinematic conservative handle clearance; native audit still required.')
