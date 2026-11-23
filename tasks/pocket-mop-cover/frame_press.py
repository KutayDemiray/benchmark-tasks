"""Ordinary robot handle press and retained lift from a free folded frame."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gate import solve_pose, array


def press_sequence(session, frame, bodies, shift, table_top, out, robot_geometry):
    """Grasp the real handle, press the head flat, lift, return and release."""
    session.hold(1., 'settle_frame')
    joints = array(frame.root_view.get_dof_positions())[0]
    angles = np.rad2deg(joints[[frame.joint_names.index('HingeLeft'), frame.joint_names.index('HingeRight')]])
    assert abs(angles).min() > 30., 'Folded starting frame collapsed before robot contact'
    pose = bodies()[frame.body_names.index('Handle')]
    axis = Rotation.from_quat(pose[3:]).as_matrix()[:, 2]
    assert axis[2] > .98, 'Handle did not remain upright in the unsupported starting setup'
    target = pose[:3] + .10 * axis
    bridge = bodies()[frame.body_names.index('Bridge'), :3].copy()
    press = target + [0., 0., table_top - .002 - bridge[2]]
    lifted = press + [0., 0., .22]
    choices = []
    for degrees in (30., 60., 90., 120., 150., 180., -30., -60., -90., 0.):
        approach = np.array([np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees)), 0.])
        closing = np.cross(approach, axis)
        closing /= np.linalg.norm(closing)
        for sign in (1., -1.):
            rotation = np.column_stack((sign * closing, np.cross(approach, sign * closing), approach))
            q = session.action[7:13].copy()
            cost = 0.
            try:
                for point in (target-.075*approach+[0,0,.04], target-.075*approach,
                              target, press, lifted, lifted+[0,.10,0]):
                    goal, _, _ = solve_pose(point-shift, rotation, q, 'right')
                    cost += float(np.sum((goal-q)**2))
                    q = goal
                choices.append((cost, degrees, sign, rotation, approach))
            except AssertionError:
                pass
    assert choices, 'No reachable folded-frame press and lift path'
    cost, degrees, sign, rotation, approach = min(choices, key=lambda item: item[0])
    (out/'handle-grasp-selection.json').write_text(json.dumps(dict(
        target_m=target.tolist(), press_target_m=press.tolist(), handle_axis=axis.tolist(),
        approach_degrees=degrees, closing_sign=sign, cost=cost,
        folded_start_angle_deg=angles.tolist()), indent=2))
    pre = target - .075 * approach
    standoff = target + [-.08, -.22, .04]
    from robot_clearance import certify_handle_approach
    certificate=certify_handle_approach(session.action[7:14],
        [(standoff,3.,True),(pre+[0,0,.04],2.,False),(pre,2.,False)],rotation,
        robot_geometry,pose[:3]+.25*axis,axis,shift)
    (out/'approach-clearance.json').write_text(json.dumps(certificate,indent=2))
    session.move('right', standoff, rotation, .6, 3., 'approach_handle_side', joint=True)
    session.move('right', pre+[0,0,.04], rotation, .6, 2., 'approach_handle')
    session.move('right', pre, rotation, .6, 2., 'align_handle')
    joints=array(frame.root_view.get_dof_positions())[0]
    assert abs(np.rad2deg(joints[frame.joint_names.index('HingeLeft')]))>30., 'Frame moved before intended contact'
    session.move('right', target, rotation, .6, 2., 'enter_handle')
    session.move('right', target, rotation, .08, 1., 'grasp_handle')
    session.hold(1., 'verify_handle_grasp')
    session.move('right', press, rotation, .08, 4., 'press_frame_open')
    session.hold(2., 'verify_frame_locked')
    session.move('right', lifted, rotation, .08, 3., 'lift_flat_frame')
    session.hold(2., 'hold_flat_frame')
    session.move('right', lifted+[0,.10,0], rotation, .08, 2., 'translate_frame')
    session.move('right', lifted, rotation, .08, 2., 'return_frame')
    session.hold(2., 'retention')
    place = press + [0., 0., .008]
    session.move('right', place, rotation, .08, 3., 'set_down_frame')
    session.move('right', place, rotation, .6, 1., 'release_handle')
    session.move('right', place-.08*approach+[0,0,.08], rotation, .6, 2., 'withdraw')
    session.hold(3., 'released')
