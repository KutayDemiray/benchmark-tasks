"""Ordinary stock YAM actions, native recordings and the three stock cameras."""
import json
import subprocess
from pathlib import Path

import numpy as np

from robot_gate import array,solve_pose


class RobotSession:
    """Record an initialized Arena scene without authoring physics during actions."""
    def __init__(self,env,out,request,observe):
        from yam_arena.constants import READY
        self.env=env
        self.base=env.unwrapped
        self.out=Path(out)
        self.request=request
        from station_coordinates import TRANSLATION_S_W
        self.shift=TRANSLATION_S_W.copy() if request.get("table_origin",False) else np.zeros(3)
        self.observe=observe
        self.action=np.r_[READY,1.,READY,1.]
        self.rows=[]
        self.plan=[]
        self.writers={}
        self.log=(self.out/'steps.jsonl').open('w',buffering=1)
        self.chunks=self.out/'physics-chunks'
        self.chunks.mkdir()
        from chunk_writer import ChunkWriter
        self.chunk_writer=ChunkWriter(self.chunks)
        self.pending={}
        self.physics_steps=0
        self.original_step=self.base.sim.step
        def recorded(*args,**kwargs):
            result=self.original_step(*args,**kwargs)
            for name,data in observe(True).items():
                self.pending.setdefault(name,[]).append(np.asarray(data).copy())
            self.physics_steps+=1
            if self.physics_steps%request['physics_hz']==0:self.flush()
            return result
        self.base.sim.step=recorded
        from PIL import Image
        images, measured = self.public_snapshot()
        for name, rgb in images.items():
            Image.fromarray(rgb).save(self.out/(name+'-initial.png'))
        Image.fromarray(images['top_cam']).save(self.out/'setup.png')
        (self.out/'initial-public-state.json').write_text(json.dumps(dict(
            measured_joint_pos=measured.tolist(), camera_shapes={name:list(rgb.shape) for name,rgb in images.items()},
            action_count=0, physics_steps_since_reset=0),indent=2))

    def flush(self):
        if self.pending:
            self.chunk_writer.submit(self.pending,self.physics_steps)
            self.pending.clear()

    def step(self,target,phase):
        import torch
        from PIL import Image
        from evaluation_contract import process_action, vector
        assert len(self.rows)<round(self.request['seconds']*20)
        raw_target=vector(target)
        target=process_action(raw_target)
        self.action=target.copy()
        self.env.step(torch.tensor(target[None],device=self.base.device,dtype=torch.float32))
        row=dict(step=len(self.rows),time_s=(len(self.rows)+1)/20,phase=phase,
            robot_action=target.tolist(),requested_robot_action=raw_target.tolist(),measured_joint_pos=self.measured_joint_pos().tolist(),
            score=0.,full_task_success=False,**self.observe(False))
        self.rows.append(row)
        self.log.write(json.dumps(row)+'\n')
        for camera in ('top_cam','left_cam','right_cam'):
            rgb=array(self.base.scene.sensors[camera].data.output['rgb'])[0,:,:,:3].astype(np.uint8)
            if camera not in self.writers:
                h,w=rgb.shape[:2]
                self.writers[camera]=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{w}x{h}','-r','20','-i','-','-c:v','libx264','-threads','2','-preset','ultrafast',
                    '-crf','23','-pix_fmt','yuv420p',str(self.out/(camera+'.mp4'))],stdin=subprocess.PIPE)
            self.writers[camera].stdin.write(rgb.tobytes())
            if len(self.rows)%20==0:Image.fromarray(rgb).save(self.out/(camera+'-latest.png'))
        if len(self.rows)%20==0:
            (self.out/'heartbeat.json').write_text(json.dumps(row,indent=2))
            print(json.dumps(row),flush=True)

    def measured_joint_pos(self):
        from yam_arena.actions import bimanual_joint_pos_14
        return array(bimanual_joint_pos_14(self.base))[0]

    def public_snapshot(self):
        from evaluation_contract import CAMERAS
        images={name:array(self.base.scene.sensors[name].data.output['rgb'])[0,:,:,:3].astype(np.uint8)
                for name in CAMERAS}
        return images,self.measured_joint_pos()

    def hold(self,seconds,phase):
        for _ in range(round(seconds*20)):self.step(self.action,phase)

    def move(self,side,position,rotation,grip,seconds,phase,joint=False):
        from yam_arena import ik,station
        offset=0 if side=='left' else 7
        q0=self.action[offset:offset+6].copy()
        p0,r0=ik.tcp_pose(q0,station.BASES[side])
        p0=p0+self.shift
        position=np.asarray(position)
        qgoal,pe,re=solve_pose(position-self.shift,rotation,q0,side)
        self.plan.append(dict(phase=phase,side=side,position=position.tolist(),rotation=rotation.tolist(),
            grip=grip,seconds=seconds,ik_error_m=pe,ik_error_rad=re))
        (self.out/'controller-plan.json').write_text(json.dumps(self.plan,indent=2))
        count=round(seconds*20)
        g0=self.action[offset+6]
        q=q0.copy()
        for k in range(1,count+1):
            s=ik.smoothstep(k/count)
            if joint:q=q0+s*(qgoal-q0)
            else:
                p=p0+s*(position-p0)
                r=ik.slerp_matrix(r0,rotation,s)
                q,pe,re=ik.solve(p-self.shift,ik.matrix_to_quat_xyzw(r),q,base_pos=station.BASES[side],iters=120)
                if pe>=.004 or re>=.04:
                    q,pe,re=solve_pose(p-self.shift,r,q,side)
                assert pe<.004 and re<.04,f'Unreachable path {phase}: {pe}, {re}'
            command=self.action.copy()
            command[offset:offset+6]=q
            command[offset+6]=g0+s*(grip-g0)
            self.step(command,phase)

    def move_both(self,targets,seconds,phase):
        """Move both wrists together through ordinary 14D joint commands.

        Args:
            targets: Map each arm name to its position, rotation and opening.
            seconds: Duration of the shared Cartesian interpolation.
            phase: Recording label; it does not affect physics or scoring.
        """
        from yam_arena import ik,station
        assert set(targets)=={'left','right'}
        starts={}
        for side,(position,rotation,grip) in targets.items():
            offset=0 if side=='left' else 7
            q=self.action[offset:offset+6].copy()
            p,r=ik.tcp_pose(q,station.BASES[side])
            _,pe,re=solve_pose(np.asarray(position)-self.shift,rotation,q,side)
            starts[side]=(p+self.shift,r,q,self.action[offset+6])
            self.plan.append(dict(phase=phase,side=side,position=np.asarray(position).tolist(),
                rotation=np.asarray(rotation).tolist(),grip=grip,seconds=seconds,
                synchronized=True,ik_error_m=pe,ik_error_rad=re))
        (self.out/'controller-plan.json').write_text(json.dumps(self.plan,indent=2))
        count=round(seconds*20)
        for k in range(1,count+1):
            s=ik.smoothstep(k/count)
            command=self.action.copy()
            for side,(position,rotation,grip) in targets.items():
                offset=0 if side=='left' else 7
                p0,r0,q,g0=starts[side]
                p=p0+s*(np.asarray(position)-p0)
                r=ik.slerp_matrix(r0,rotation,s)
                q,pe,re=ik.solve(p-self.shift,ik.matrix_to_quat_xyzw(r),q,
                    base_pos=station.BASES[side],iters=120)
                if pe>=.004 or re>=.04:
                    q,pe,re=solve_pose(p-self.shift,r,q,side)
                assert pe<.004 and re<.04,f'Unreachable {side} path {phase}: {pe}, {re}'
                starts[side]=(p0,r0,q,g0)
                command[offset:offset+6]=q
                command[offset+6]=g0+s*(grip-g0)
            self.step(command,phase)

    def close(self):
        self.base.sim.step=self.original_step
        try:
            self.flush()
            self.chunk_writer.close()
        finally:
            self.log.close()
            for writer in self.writers.values():
                writer.stdin.close()
                writer.wait()
