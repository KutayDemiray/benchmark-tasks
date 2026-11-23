"""Open the free folding head using contact grasps on its handle and inner rail."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gate import array,solve_pose


def transform(pose):
    """Return an active world-from-body transform from a native xyzw pose."""
    value=np.eye(4)
    value[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix()
    value[:3,3]=pose[:3]
    return value


def wrist_transform(session,side):
    """Measure a wrist transform using the stock robot's joint positions."""
    from yam_arena import ik,station
    offset=0 if side=='left' else 7
    q=session.measured_joint_pos()[offset:offset+6]
    point,rotation=ik.tcp_pose(q,station.BASES[side])
    value=np.eye(4)
    value[:3,3]=point+session.shift
    value[:3,:3]=rotation
    return value


def handle_guidance_target(T_W_B,T_W_H,T_W_TCP,handle_y):
    """Rotate the held handle inward from its stop through an ordinary wrist goal.

    The native HandleY axis passes through the handle body's origin. Rotating
    the grasp around that point preserves the contact grasp and leaves room
    for its observed angular compliance, without commanding a passive joint.
    """
    rotation=np.eye(4)
    rotation[:3,:3]=Rotation.from_euler('y',np.deg2rad(70.)-handle_y).as_matrix()
    return np.linalg.inv(T_W_B)@T_W_H@rotation@np.linalg.inv(T_W_H)@T_W_TCP


def frame_fold_degrees(current):
    """Measure the largest wing fold relative to the bridge."""
    angles=[Rotation.from_matrix(current['Bridge'][:3,:3].T@current[side][:3,:3]).as_euler('xyz')[1]
            for side in ('Left','Right')]
    return float(np.max(abs(np.rad2deg(angles))))


def handle_pose(T_W_B, x, y):
    """Return the handle pose from the qualified frame's two passive gimbal angles."""
    rx=Rotation.from_euler('x',x).as_matrix()
    T_B_H=np.eye(4)
    T_B_H[:3,:3]=rx@Rotation.from_euler('y',y).as_matrix()
    T_B_H[:3,3]=[.030,0.,.052]+rx@np.array([0.,0.,.010])
    return T_W_B@T_B_H


def upright_handle_for_press(session,T_W_B,T_W_H,T_W_TCP):
    """Move the held handle along its gimbal arc using ordinary wrist commands."""
    relative=T_W_B[:3,:3].T@T_W_H[:3,:3]
    start=np.array([np.arctan2(relative[2,1],relative[1,1]),
                    np.arctan2(relative[0,2],relative[0,0])])
    reconstructed=handle_pose(T_W_B,*start)
    assert np.max(abs(reconstructed-T_W_H))<5e-5, 'Measured handle does not match the qualified passive chain'
    local_grasp=np.linalg.inv(T_W_H)@T_W_TCP
    goal=np.deg2rad([0.,35.])
    for index in range(1,9):
        angles=start+(goal-start)*index/8
        target=handle_pose(T_W_B,*angles)@local_grasp
        session.move('right',target[:3,3],target[:3,:3],.08,.25,'orient_handle_for_press')
    (session.out/'closing-handle-plan.json').write_text(json.dumps(dict(
        initial_angles_deg=np.rad2deg(start).tolist(),target_angles_deg=[0.,35.],
        duration_s=2.,ordinary_wrist_commands_only=True,
        reason='Leave wrist joint margin while transmitting closing pressure through the real handle.'),indent=2))
    return np.linalg.inv(T_W_B)@target


def finish_press(session,press,read_frame=None):
    """Apply bounded ordinary wrist pressure, stopping when the head reaches flat."""
    records=[]
    for index in range(1,8):
        angle=None if read_frame is None else frame_fold_degrees(read_frame())
        if angle is not None and angle<=1.:
            break
        depth=.005*index
        session.move('right',press[:3,3]-[0,0,depth],press[:3,:3],.08,.4,'finish_frame_press')
        records.append(dict(extra_command_depth_m=depth,fold_before_deg=angle,
            fold_after_deg=None if read_frame is None else frame_fold_degrees(read_frame())))
    (session.out/'closing-press.json').write_text(json.dumps(dict(
        ordinary_wrist_commands_only=True,unchanged_stock_actuator_limits=True,
        maximum_extra_depth_m=.035,maximum_extra_duration_s=2.8,records=records),indent=2))


def free_retreat_rotation(session,position,side='left'):
    """Choose a reachable wrist orientation after the fingers clear the rail."""
    candidates=[]
    offset=0 if side=='left' else 7
    q0=session.action[offset:offset+6]
    for azimuth,elevation in ((-90.,45.),(-60.,45.),(-120.,45.),(-90.,60.),(-60.,60.)):
        if side=='right':azimuth=-azimuth
        az,el=np.deg2rad([azimuth,elevation])
        approach=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),-np.sin(el)])
        closing=np.cross(approach,[0.,0.,1.]);closing/=np.linalg.norm(closing)
        for sign in (-1.,1.):
            rotation=np.column_stack((sign*closing,np.cross(approach,sign*closing),approach))
            try:
                q,_,_=solve_pose(position-session.shift,rotation,q0,side)
                candidates.append((float(np.sum((q-q0)**2)),rotation))
            except AssertionError:
                pass
    assert candidates, 'No reachable free hand retreat'
    return min(candidates,key=lambda value:value[0])[1]


def reopen_sequence(session,frame,bodies,shift,out,right_rotation,lift_height):
    """Guide a grasped rail toward flat, then press and test the passive latch."""
    poses=bodies()
    q=array(frame.root_view.get_dof_positions())[0]
    angle=-float(q[frame.joint_names.index('HingeLeft')])
    assert np.deg2rad(15.)<angle<np.deg2rad(85.), 'Need an unlocked, partly folded frame'
    T_W_L=transform(poses[frame.body_names.index('Left')])
    from rail_clearance import RailClearance
    robot_geometry=json.loads((out/'robot-cooked-colliders.json').read_text())['left']
    clearance=RailClearance(json.loads((out/'frame-geometry.json').read_text()),poses,frame.body_names,robot_geometry,shift)
    candidates=[]
    checked=[]
    for rail_x,rail_y in ((-.025,.046),(-.046,.046),(-.065,.046),
                          (-.025,-.046),(-.046,-.046),(-.065,-.046)):
        point=T_W_L[:3,3]+T_W_L[:3,:3]@np.array([rail_x,rail_y,0.])
        rail_rotation=T_W_L[:3,:3]
        approach=-np.sign(rail_y)*rail_rotation[:,1]
        for sign in (-1.,1.):
            closing=sign*rail_rotation[:,2]
            rotation=np.column_stack((closing,np.cross(approach,closing),approach))
            pre=point-.08*approach
            q0=session.action[:6].copy()
            cost=0.
            try:
                for target in (pre+[0,0,.12],pre,point):
                    q1,_,_=solve_pose(target-shift,rotation,q0,'left')
                    cost+=float(np.sum((q1-q0)**2));q0=q1
                hits=clearance.contacts(q0,.25)
                checked.append(dict(rail_x_m=rail_x,rail_y_m=rail_y,sign=sign,hits=hits))
                if not hits:candidates.append((cost,rail_x,rail_y,sign,point,pre,rotation))
            except AssertionError:
                pass
    (out/'rail-clearance-preflight.json').write_text(json.dumps(checked,indent=2))
    assert candidates, 'No reachable collision-free ordinary inner-rail grasp'
    _,rail_x,rail_y,sign,point,pre,rotation=min(candidates,key=lambda value:value[0])
    (out/'rail-grasp-plan.json').write_text(json.dumps(dict(rail_x_m=rail_x,rail_y_m=rail_y,
        grasp='Pinch the 6 mm rail thickness from its exposed outer side',grasp_command=0.,
        wrist_pitch_relative_to_wing_deg=0.,closing_sign=sign,target_m=point.tolist(),
        starting_fold_angle_deg=float(np.rad2deg(angle)),ordinary_actions_only=True),indent=2))
    session.move('left',pre+[0,0,.12],rotation,.25,3.,'approach_above_inner_rail',joint=True)
    session.move('left',pre,rotation,.25,2.,'approach_inner_rail')
    current=bodies()[frame.body_names.index('Left')]
    current_point=current[:3]+Rotation.from_quat(current[3:]).as_matrix()@np.array([rail_x,rail_y,0.])
    assert np.linalg.norm(current_point-point)<.008, 'Frame moved during the nominally clear rail approach'
    session.move('left',point,rotation,.25,2.,'enter_inner_rail')
    session.move('left',point,rotation,0.,1.,'grasp_inner_rail')
    session.hold(1.,'verify_inner_rail_grasp')
    poses=bodies()
    T_W_B=transform(poses[frame.body_names.index('Bridge')])
    T_W_L=transform(poses[frame.body_names.index('Left')])
    T_W_H=transform(poses[frame.body_names.index('Handle')])
    T_L_TCP=np.linalg.inv(T_W_L)@wrist_transform(session,'left')
    T_H_TCP=np.linalg.inv(T_W_H)@wrist_transform(session,'right')
    T_B_TCP=np.linalg.inv(T_W_B)@wrist_transform(session,'right')
    q=array(frame.root_view.get_dof_positions())[0]
    if session.cover_center is not None:
        handle_y=float(q[frame.joint_names.index('HandleY')])
        T_B_TCP=handle_guidance_target(T_W_B,T_W_H,wrist_transform(session,'right'),handle_y)
        (out/'handle-guidance-plan.json').write_text(json.dumps(dict(
            initial_handle_y_deg=float(np.rad2deg(handle_y)),planned_handle_y_deg=70.,
            ordinary_wrist_target_only=True,passive_joint_limit_deg=85.,
            reason='Leave room below the passive stop for measured angular compliance of the handle grasp.'),indent=2))
    angle=-float(q[frame.joint_names.index('HingeLeft')])
    rail_center=np.array([rail_x,rail_y,0.])
    assert np.linalg.norm(T_L_TCP[:3,3]-rail_center)<.025, 'Rail moved before the contact grasp'
    def read_frame():
        current=bodies()
        return {name:transform(current[frame.body_names.index(name)]) for name in ('Bridge','Left','Right','Handle')}
    def measure_grasp(side):
        # Track a marker fixed to each held body, at the original grasp pose.
        # This includes slip/compliance inside the fingers, which joint FK
        # cannot observe. Only the robot receives commands.
        current=read_frame()
        return current['Left']@T_L_TCP if side=='left' else current['Handle']@T_H_TCP
    guide_and_place(session,T_W_B,T_L_TCP,T_B_TCP,angle,rail_y,lift_height,shift,
        measure_wrist=lambda side:wrist_transform(session,side),read_frame=read_frame,
        measure_grasp=measure_grasp,handle_grasp=T_H_TCP)


def guide_and_place(session,T_W_B,T_L_TCP,T_B_TCP,angle,rail_y,lift_height,shift,
                    measure_wrist=None,read_frame=None,measure_grasp=None,handle_grasp=None):
    """Execute consistent two-hand frame goals and the final ordinary retention motion."""
    yaw=Rotation.from_matrix(T_W_B[:3,:3]).as_euler('xyz')[2]
    upright=Rotation.from_euler('z',yaw).as_matrix()
    from yam_arena import ik,station
    table_top=station.TABLE_TOP+shift[2]
    tracker=None
    if session.cover_center is not None:
        from wrist_tracking import MeasuredWristGuide
        tracker=MeasuredWristGuide(session,measure_grasp or measure_wrist,
            measure_robot=measure_wrist if measure_grasp is not None else None,
            measurement='native_held_body_markers' if measure_grasp is not None else 'robot_joint_fk')
        # The separate cover is approached only after both ordinary contact
        # grasps are established. These are wrist goals, not object constraints.
        theta=np.deg2rad(45.)
        T_B_L=np.eye(4)
        T_B_L[:3,:3]=Rotation.from_euler('y',-theta).as_matrix()
        T_B_L[:3,3]=[-.008,0.,0.]
        carry=T_W_B.copy()
        carry[:3,:3]=np.eye(3)
        # The passive handle gimbal lets the far wing hang below its nominal
        # symmetric pose during a carry. Keep clearance for that observed sag.
        carry_clearance=.11
        carry[2,3]=table_top+.192*np.sin(theta)+.003*np.cos(theta)+carry_clearance
        left=carry@T_B_L@T_L_TCP
        right=carry@T_B_TCP
        targets={'left':(left[:3,3],left[:3,:3],0.),'right':(right[:3,3],right[:3,:3],.08)}
        tracker.move(targets,3.,'prepare_cover_approach')
        tracker.settle(targets,'level_frame_before_carry')
        carry[:2,3]=session.cover_center[:2]
        left=carry@T_B_L@T_L_TCP
        right=carry@T_B_TCP
        targets={'left':(left[:3,3],left[:3,:3],0.),'right':(right[:3,3],right[:3,:3],.08)}
        tracker.move(targets,3.,'carry_frame_to_cover')
        tracker.settle(targets,'level_frame_over_cover')
        if read_frame is not None:
            current=read_frame()
            tips=[]
            for side,sign in (('Left',-1.),('Right',1.)):
                pose=current[side]
                tips.append(pose[:3,3]+pose[:3,:3]@[sign*.192,0.,-.003])
            proof=dict(tips_m=[tip.tolist() for tip in tips],
                height_difference_m=float(abs(tips[0][2]-tips[1][2])),
                head_center_error_m=float(np.linalg.norm(current['Bridge'][:2,3]-carry[:2,3])))
            (session.out/'pre-insertion-alignment.json').write_text(json.dumps(proof,indent=2))
            assert proof['height_difference_m']<.006 and proof['head_center_error_m']<.01, 'Loaded head is not aligned for pocket entry'
        # Match the qualified blade's 1 mm lower-face height. Measure robot
        # tracking throughout descent, before accumulated sag can move the cover.
        for _ in range(10):
            carry[2,3]-=(carry_clearance-.001)/10
            left=carry@T_B_L@T_L_TCP
            right=carry@T_B_TCP
            tracker.move({'left':(left[:3,3],left[:3,:3],0.),
                          'right':(right[:3,3],right[:3,:3],.08)},.3,'align_pocket_mouths')
        T_W_B=carry
        angle=theta
        upright=np.eye(3)
    target_angle=np.deg2rad(15.)
    # Each target comes from a consistent frame geometry. Only ordinary wrist
    # commands are applied; neither the bridge nor any passive joint is driven.
    for index in range(1,13):
        fraction=index/12
        theta=angle+fraction*(target_angle-angle)
        T_W_B_goal=T_W_B.copy()
        T_W_B_goal[:3,:3]=ik.slerp_matrix(T_W_B[:3,:3],upright,fraction)
        height=table_top+.192*np.sin(theta)+.003*np.cos(theta)+.001
        # During insertion the noses must slide along the base sheet into the
        # mouths. Blending the root height leaves them above the pocket roofs.
        T_W_B_goal[2,3]=height if session.cover_center is not None else (1-fraction)*T_W_B[2,3]+fraction*height
        T_B_L=np.eye(4)
        T_B_L[:3,:3]=Rotation.from_euler('y',-theta).as_matrix()
        T_B_L[:3,3]=[-.008,0.,0.]
        left=T_W_B_goal@T_B_L@T_L_TCP
        right=T_W_B_goal@T_B_TCP
        targets={'left':(left[:3,3],left[:3,:3],0.),'right':(right[:3,3],right[:3,:3],.08)}
        if tracker is None:session.move_both(targets,.5,'guide_frame_open')
        else:tracker.move(targets,.5,'guide_frame_open')
    session.hold(1.,'hold_guided_frame')
    if tracker is not None:
        point,rotation,_=tracker.last_targets['left']
        left[:3,3]=point;left[:3,:3]=rotation
    session.move('left',left[:3,3],left[:3,:3],.25,1.,'release_inner_rail')
    outward=np.sign(rail_y)*upright[:,1]
    # Slide the fingers out of the rail before raising them. A straight upward
    # withdrawal can scoop the rail with the lower finger even with open jaws.
    session.move('left',left[:3,3]+.03*outward,left[:3,:3],.25,1.5,'withdraw_from_inner_rail')
    outside=.12*outward+[0,0,.15]
    retreat=left[:3,3]+outside
    free_rotation=free_retreat_rotation(session,retreat)
    session.move('left',retreat,free_rotation,.25,3.,'clear_inner_rail')
    T_W_B_goal[2,3]=table_top+.002
    press=T_W_B_goal@T_B_TCP
    session.move('right',press[:3,3],press[:3,:3],.08,3.,'press_frame_flat')
    # Account for measured closing deflection through ordinary wrist pressure.
    # Stop on the observed fold; no passive joint or latch is commanded.
    if session.cover_center is not None:
        from bridge_press import press_bridge
        bridge=read_frame()['Bridge'] if read_frame is not None else T_W_B_goal.copy()
        press_bridge(session,bridge,read_frame)
    else:
        finish_press(session,press,read_frame)
    # The cover press already holds for .5 s and clears its hand over 3 s.
    # Allocate 1.8 s of this former dwell to the bounded central press while
    # preserving the full controller's 2,400-action ceiling.
    session.hold(.2 if session.cover_center is not None else 2.,'hold_pressed_frame')
    if read_frame is not None:
        assert frame_fold_degrees(read_frame())<=3., 'Frame did not close; folded lift/set-down would not test retention'
    lifted=press[:3,3]+[0,0,lift_height]
    session.move('right',lifted,press[:3,:3],.08,3.,'lift_flat_frame')
    session.hold(2.,'hold_flat_frame')
    session.move('right',lifted+[-.12,0,0],press[:3,:3],.08,2.,'translate_frame')
    session.move('right',lifted,press[:3,:3],.08,2.,'return_frame')
    session.hold(2.,'retention')
    if getattr(session, 'request', {}).get('completion_mode', 'released') == 'handle-held':
        assert session.cover_center is not None
        # Versioned endpoint: ordinary handle grasp remains active. The full
        # native recording still requires independent geometry and contact audits.
        return
    if session.cover_center is not None:
        from supported_release import place_supported
        position,rotation=ik.tcp_pose(session.action[7:13],station.BASES['right'])
        wrist=np.eye(4);wrist[:3,3]=position+shift;wrist[:3,:3]=rotation
        if read_frame is None:
            bridge=T_W_B_goal.copy();bridge[2,3]+=lift_height
            handle=wrist@np.linalg.inv(handle_grasp)
        else:
            current=read_frame();bridge=current['Bridge'];handle=current['Handle']
            wrist=measure_wrist('right')
        place_supported(session,press[:3,3]+[0,0,.003],press[:3,:3],
                        bridge,handle,wrist,table_top,read_frame,measure_wrist)
        return
    place=press[:3,3]+[0,0,.003]
    session.move('right',place,press[:3,:3],.08,3.,'set_down_frame')
    session.move('right',place,press[:3,:3],.6,1.,'release_handle')
    session.move('right',place-.08*press[:3,2]+[0,0,.08],press[:3,:3],.6,2.,'withdraw')
    session.hold(6. if session.cover_center is not None else 3.,'released')
