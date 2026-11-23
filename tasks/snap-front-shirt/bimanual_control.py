"""Bounded ordinary-action pilot: left fabric support and right snap pressing."""

import numpy as np

from tabletop_control import TablePressController

# Contact locations/orientations are controller hypotheses, not physical changes.
PILOTS = {
    1: dict(x=.36, y=.16, tilt_degrees=-45., lowest_tip_z=.7505),
    2: dict(x=.36, y=.18, tilt_degrees=-45., lowest_tip_z=.7505),
    3: dict(x=.28, y=.22, tilt_degrees=-45., lowest_tip_z=.7505),
}


def support_pose(revision):
    """Find the stock-joint pose whose lowest finger surface meets the cloth."""
    from scipy.spatial.transform import Rotation
    from yam_arena import ik
    from yam_arena.analysis import fk_link_poses_world, world_hulls
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.urdf import quat_xyzw_to_matrix
    settings = PILOTS[revision]
    rotation = Rotation.from_euler('x',settings['tilt_degrees'],degrees=True).as_matrix()
    rotation = rotation @ quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
    quaternion = ik.matrix_to_quat_xyzw(rotation)
    target = np.array([settings['x'],settings['y'],.78])
    q = np.asarray(READY).copy()
    for _ in range(3):
        q,error,angle = ik.solve(target,quaternion,q,base_pos=LEFT_BASE_POS,iters=200)
        assert error<.0002 and angle<.002, 'Support pose is unreachable'
        joints = dict(zip(ARM_JOINTS,q,strict=True))
        joints.update({name:0. for name in FINGER_JOINTS})
        hulls = world_hulls(fk_link_poses_world(joints,LEFT_BASE_POS))
        bottom = min(h.vertices[:,2].min() for name,h in hulls if name.startswith('tip_'))
        target[2] += settings['lowest_tip_z']-bottom
    return target,rotation


class BimanualPressController:
    """Pin exposed lower fabric, press one snap, then withdraw both arms."""

    def __init__(self,left_q,left_tcp,right_q,right_tcp,cfg,revision=1):
        from yam_arena import ik
        from yam_arena.constants import LEFT_BASE_POS
        self.ik,self.base,self.cfg = ik,LEFT_BASE_POS,cfg
        self.revision = revision
        self.q = np.asarray(left_q).copy()
        self.start = np.asarray(left_tcp).copy()
        self.start_rotation = ik.tcp_pose(self.q,self.base)[1]
        self.target,self.rotation = support_pose(revision)
        self.hover = self.target+np.array([0.,0.,.10])
        self.right_q = np.asarray(right_q).copy()
        self.right_tcp = np.asarray(right_tcp).copy()
        self.press = None
        self.release_started = None
        self.completed = False
        self.abort_reason = None

    def action(self,time_s,left_tcp,right_tcp,state):
        if time_s<3.:
            phase='support_approach'
            f=self.ik.smoothstep(time_s/3.)
            target=(1-f)*self.start+f*self.hover
            rotation=self.ik.slerp_matrix(self.start_rotation,self.rotation,f)
        elif time_s<6.:
            phase='support_lower'
            f=self.ik.smoothstep((time_s-3.)/3.)
            target=(1-f)*self.hover+f*self.target
            rotation=self.rotation
        else:
            phase='support_hold'
            target=self.target.copy();rotation=self.rotation
        control=dict(phase=phase,target_pair=0,completed=False,abort_reason=None)
        if time_s>=6.:
            if self.press is None:
                self.press=TablePressController(self.right_q,right_tcp,self.cfg,targets=(0,),press_side='right')
                from yam_arena.constants import RIGHT_BASE_POS,READY
                self.press.final_view_position,self.press.final_view_rotation=self.ik.tcp_pose(READY,RIGHT_BASE_POS)
                self.press.retreat_height_m=.09
            action,control=self.press.action(time_s-6.,right_tcp,state)
            self.right_q=np.asarray(action[7:13]).copy()
            if self.press.phase in ('final_view','released_observation') and self.release_started is None:
                self.release_started=time_s
                self.release_start=np.asarray(left_tcp).copy()
            if self.release_started is not None:
                phase='support_withdraw'
                f=self.ik.smoothstep((time_s-self.release_started)/2.)
                target=(1-f)*self.release_start+f*(self.target+np.array([0.,0.,.12]))
                if time_s-self.release_started>=2.:
                    phase='support_released'
        self.q,error,angle=self.ik.solve(target,self.ik.matrix_to_quat_xyzw(rotation),self.q,
                                       base_pos=self.base,iters=150)
        self.abort_reason=control['abort_reason']
        self.completed=bool(control['completed'] and phase=='support_released')
        control.update(completed=self.completed,support_phase=phase,support_revision=self.revision,
            support_target_tcp_m=target,support_ik_error_m=error,support_ik_angle_rad=angle,
            support_arm='left',pressing_arm='right',scope='one_snap_bimanual_diagnostic')
        return np.r_[self.q,0.,self.right_q,0. if self.press is not None else 1.],control
