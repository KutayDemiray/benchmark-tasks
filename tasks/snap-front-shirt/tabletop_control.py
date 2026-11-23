"""Ordinary stock-action contact feedback for sequential tabletop shirt snaps."""

import numpy as np

from robot_scoring import snap_seated


class TablePressController:
    """Approach, frictionally align, press, withdraw and observe a free placket."""

    def __init__(self,q,tcp,cfg,targets=(0,),observer=False,press_side='left'):
        from yam_arena import ik
        from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS
        from yam_arena.demo import DOWN_GRIP
        from yam_arena.urdf import quat_xyzw_to_matrix
        assert press_side in ('left','right')
        assert not observer or press_side=='left', 'Observer pose is defined for left-arm pressing'
        self.press_side=press_side
        self.final_view_position=np.array([.18,.25 if press_side=='left' else -.25,1.00])
        self.ik=ik;self.base=LEFT_BASE_POS if press_side=='left' else RIGHT_BASE_POS;self.q=np.asarray(q).copy()
        self.start=np.asarray(tcp).copy();self.start_rotation=ik.tcp_pose(self.q,self.base)[1]
        self.rotation=quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
        self.retreat_height_m=.12
        self.final_view_rotation=self.rotation.copy()
        self.cfg=cfg;self.targets=list(targets);self.index=0
        self.phase='approach';self.phase_started=0.;self.xy=None;self.press_z=None
        self.seated=0;self.aligned=0;self.abort_reason=None;self.completed=False
        self.retreat_start=None;self.previous_target=self.start.copy()
        self.finished_pairs=[]
        self.observer_q=None
        if observer:
            from yam_arena.constants import RIGHT_BASE_POS,READY,WRIST_CAM_POS,WRIST_CAM_ROT_XYZW,TCP_OFFSET
            camera_position=np.array([.36,-.12,1.08]);look_at=np.array([.36,.12,.77])
            z=look_at-camera_position;z/=np.linalg.norm(z)
            x=np.array([1.,0.,0.]);y=np.cross(z,x)
            R_W_C=np.column_stack((x,y,z))
            R_W_G=R_W_C@quat_xyzw_to_matrix(np.array(WRIST_CAM_ROT_XYZW)).T
            tcp_position=camera_position+R_W_G@(np.array(TCP_OFFSET)-WRIST_CAM_POS)
            self.observer_q,error,angle=ik.solve(tcp_position,ik.matrix_to_quat_xyzw(R_W_G),
                np.array(READY),base_pos=RIGHT_BASE_POS,iters=200)
            assert error<.0001 and angle<.001,'observer_pose_unreachable'

    def transition(self,phase,time_s):
        self.phase=phase;self.phase_started=time_s

    def action(self,time_s,measured_tcp,state):
        from yam_arena.constants import READY
        target_index=self.targets[self.index]
        m=np.asarray(state['hardware'][f'male{target_index}'])
        s=np.asarray(state['hardware'][f'socket{target_index}'])
        force=state.get('snap_robot_contact_norm_N',{}).get(f'male{target_index}',0.)
        delta=s[:2]-m[:2];lateral=float(np.linalg.norm(delta))
        seated=snap_seated(m,s,self.cfg)
        h=self.cfg['tabletop']['table_top_m']
        hover=np.r_[m[:2],h+.065]
        duration=time_s-self.phase_started
        rot=self.rotation
        if self.phase=='approach':
            f=self.ik.smoothstep(min(1.,duration/3.))
            target=(1-f)*self.start+f*hover
            rot=self.ik.slerp_matrix(self.start_rotation,self.rotation,f)
            if duration>=3.:
                self.xy=m[:2].copy();self.press_z=float(target[2]);self.transition('touch',time_s)
        elif self.phase=='touch':
            self.xy=m[:2].copy()
            self.press_z=max(m[2]+.008,self.press_z-.001)
            target=np.r_[self.xy,self.press_z]
            if force>.15:self.transition('align',time_s)
            elif duration>7.:
                self.abort_reason='no_pressing_contact';self.transition('withdraw',time_s);self.retreat_start=target.copy()
        elif self.phase=='align':
            correction=.25*delta
            length=np.linalg.norm(correction)
            if length>.0003:correction*=.0003/length
            self.xy+=correction
            if force<.15:self.press_z-=.00015
            elif force>1.5:self.press_z+=.00015
            target=np.r_[self.xy,self.press_z]
            self.aligned=self.aligned+1 if lateral<.00045 else 0
            if self.aligned>=3 or seated:self.transition('press',time_s)
            elif duration>5.:
                self.abort_reason='placket_alignment_timeout';self.transition('withdraw',time_s);self.retreat_start=target.copy()
        elif self.phase=='press':
            self.xy+=np.clip(.12*delta,-.0001,.0001)
            if not seated and force<10.:self.press_z-=.00015
            elif force>12.:self.press_z+=.0002
            self.press_z=max(h-.005,self.press_z)
            target=np.r_[self.xy,self.press_z]
            self.seated=self.seated+1 if seated else 0
            if self.seated>=4 or duration>9.:
                if self.seated<4:self.abort_reason='snap_did_not_seat'
                self.retreat_start=target.copy();self.transition('withdraw',time_s)
        elif self.phase=='withdraw':
            if self.retreat_start is None:self.retreat_start=self.previous_target.copy()
            f=self.ik.smoothstep(min(1.,duration/2.))
            retreat=np.r_[m[:2],h+self.retreat_height_m]
            target=(1-f)*self.retreat_start+f*retreat
            if duration>=2.:
                if self.abort_reason is not None or self.index==len(self.targets)-1:
                    self.retreat_start=target.copy();self.transition('final_view',time_s)
                else:self.transition('released_observation',time_s)
        elif self.phase=='final_view':
            f=self.ik.smoothstep(min(1.,duration/2.))
            target=(1-f)*self.retreat_start+f*self.final_view_position
            rot=self.ik.slerp_matrix(self.rotation,self.final_view_rotation,f)
            if duration>=2.:self.transition('released_observation',time_s)
        else:
            final=self.abort_reason is not None or self.index==len(self.targets)-1
            target=self.final_view_position if final else np.r_[m[:2],h+self.retreat_height_m]
            if final:rot=self.final_view_rotation
            if duration>=(3.2 if final else .4):
                if not seated and self.abort_reason is None:self.abort_reason='closure_lost_after_withdrawal'
                if self.abort_reason is None:self.finished_pairs.append(target_index)
                if self.abort_reason is None and self.index+1<len(self.targets):
                    self.index+=1;self.seated=0;self.aligned=0;self.retreat_start=None
                    self.start=np.asarray(measured_tcp).copy()
                    self.start_rotation=self.ik.tcp_pose(self.q,self.base)[1]
                    self.transition('approach',time_s)
                else:self.completed=True
        if force>25. and self.abort_reason is None:
            self.abort_reason='press_force_limit';self.retreat_start=target.copy();self.transition('withdraw',time_s)
        self.q,error,angle=self.ik.solve(target,self.ik.matrix_to_quat_xyzw(rot),self.q,base_pos=self.base,iters=100)
        self.previous_target=target.copy()
        right=np.array(READY)
        if self.observer_q is not None:
            f=self.ik.smoothstep(min(1.,time_s/3.))
            right=(1-f)*right+f*self.observer_q
        action=np.r_[self.q,0.,right,1.] if self.press_side=='left' else np.r_[right,1.,self.q,0.]
        return action,dict(phase=self.phase,target_pair=target_index,target_tcp_m=target,
            ik_error_m=error,ik_angle_rad=angle,lateral_error_m=lateral,normal_contact_norm_N=force,
            seated=seated,completed=self.completed,finished_pairs=self.finished_pairs.copy(),abort_reason=self.abort_reason)
