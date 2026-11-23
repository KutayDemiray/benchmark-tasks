"""Two ordinary grippers substitute for handle holding and pedal operation."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gate import solve_pose


def unlock_sequence(session, frame, bodies, shift, out, right_target, right_rotation, lift_height):
    """Press the physical rocker while supporting and lifting the handle."""
    from yam_arena.constants import PAD_TIP_PAST_TCP
    pose=bodies()[frame.body_names.index('Rocker')]
    point=pose[:3]+Rotation.from_quat(pose[3:]).as_matrix()@np.array([-.030,0.,.024])
    candidates=[]
    for azimuth,elevation in ((-60.,45.),(-45.,45.),(-90.,45.),(-60.,60.),(-45.,60.),
                              (-90.,60.),(-60.,30.),(-45.,30.),(-90.,30.)):
        az,el=np.deg2rad([azimuth,elevation])
        approach=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),-np.sin(el)])
        closing=np.cross(approach,[0.,0.,1.]);closing/=np.linalg.norm(closing)
        pressed=point-PAD_TIP_PAST_TCP*approach-[0.,0.,.018]
        pre=pressed+[0.,0.,.15]
        above=pressed+[0.,0.,.038]
        for sign in (1.,-1.):
            rotation=np.column_stack((sign*closing,np.cross(approach,sign*closing),approach))
            q=session.action[:6].copy();cost=0.
            try:
                for target in (pre,above,pressed):
                    goal,_,_=solve_pose(target-shift,rotation,q,'left')
                    cost+=float(np.sum((goal-q)**2));q=goal
                candidates.append((cost,azimuth,elevation,sign,rotation,pre,above,pressed))
            except AssertionError:
                pass
    assert candidates, 'No reachable physical pedal press'
    _,azimuth,elevation,sign,rotation,pre,above,pressed=min(candidates,key=lambda item:item[0])
    (out/'pedal-plan.json').write_text(json.dumps(dict(surface_point_m=point.tolist(),
        pressed_tcp_m=pressed.tolist(),pad_tip_past_tcp_m=PAD_TIP_PAST_TCP,
        azimuth_degrees=azimuth,elevation_degrees=elevation,closing_sign=sign,
        ordinary_actions_only=True),indent=2))
    session.move('left',pre,rotation,0.,3.,'approach_pedal',joint=True)
    session.move('left',above,rotation,0.,2.,'align_pedal')
    session.move('left',pressed,rotation,0.,2.,'press_pedal')
    # The handle gimbal absorbs part of wrist travel. Keep the pedal pressed
    # through a larger wrist lift so the rocker can actually leave its groove.
    session.move('right',right_target+[0,0,.10],right_rotation,.08,3.,'lift_while_pressing_pedal')
    session.hold(1.,'hold_pedal_and_handle')
    session.move('left',pre,rotation,0.,2.,'release_pedal')
    if session.request.get('robot_reopen',False):
        from robot_reopen import reopen_sequence
        session.move('left',pre+[0,0,.15],rotation,0.,2.,'clear_pedal_workspace')
        session.move('right',right_target+[0,0,.17],right_rotation,.08,3.,'lift_for_rail_grasp')
        session.hold(1.,'settle_folded_frame')
        reopen_sequence(session,frame,bodies,shift,out,right_rotation,lift_height)
        return
    session.move('right',right_target+[0,0,lift_height],right_rotation,.08,3.,'lift_after_unlock')
    session.hold(2.,'unlocked_hold')
    session.move('right',right_target,right_rotation,.08,3.,'set_down_frame')
    session.move('right',right_target,right_rotation,.6,1.,'release_handle')
    session.move('right',right_target-.08*right_rotation[:,2]+[0,0,.08],right_rotation,.6,2.,'withdraw')
    session.hold(2.,'released')
