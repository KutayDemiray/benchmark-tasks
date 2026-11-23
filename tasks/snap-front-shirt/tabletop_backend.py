"""Reusable native episode lifecycle for ordinary-action tabletop evaluation."""

import json
import time

import numpy as np

from bench import write,jsonable
from episode_timing import control_hz,max_actions
from garment_runner import Measurements,verify_scene
from policy_contract import CAMERAS,INSTRUCTION,observation,validate_action
from streaming_score import StreamingScore
from tabletop_clearance import robot_clearance,stock_collision_bounds
from tabletop_runner import table_observation,add_contact_measurements
from tabletop_scoring import task_score


class TabletopEpisode:
    """Record and score one native episode; a new process owns every reset."""

    def __init__(self, env, cfg, args, meta, destination, bundle, instruction=INSTRUCTION):
        import torch
        from yam_arena.runtime import tcp_world
        self.torch,self.tcp_world=torch,tcp_world
        self.env,self.base=env,env.unwrapped
        self.cfg,self.args,self.meta=cfg,args,meta
        self.destination,self.bundle=destination,bundle
        self.instruction=instruction
        self.hz=control_hz(cfg)
        self.rows,self.nodes=[],[]
        self.writers={}
        self.closed=False
        self.reason=None
        self.started=time.monotonic()
        env.reset()
        base=self.base
        assert base.action_manager.total_action_dim==14
        assert abs(base.step_dt-1/self.hz)<1e-9
        assert list(base.action_manager.active_terms)==['left_arm','left_gripper','right_arm','right_gripper']
        verify_scene(base,meta,destination,args.record)
        meta['robot_collision_bounds']=stock_collision_bounds()
        write(destination/'setup.json',dict(parameters=cfg,seed=args.seed,case='fasten_three',
            stage='tabletop_shirt',recorded_cameras=args.record,control_hz=self.hz,
            action_dimension=14,bundle_sha256=bundle,reset_mode='fresh_native_process',
            controller=getattr(args,'controller','none'),policy_images=[320,176]))
        write(destination/'topology.json',meta)
        self.measure=Measurements(base,meta,destination,extra_observer=table_observation(cfg),
                                  strain_witness=args.strain_witness)
        add_contact_measurements(self.measure,meta,destination)
        if getattr(args,'pilot_diagnostics',False):
            from pilot_measurements import add_pilot_measurements
            add_pilot_measurements(self.measure,meta,destination)
        self.scorer=StreamingScore(cfg,meta)
        self.measure.state_observer=self.scorer.observe_physics
        self.measure.observe()
        self.left,self.right=base.scene['left_robot'],base.scene['right_robot']
        np.savez_compressed(destination/'initial-measured-state.npz',cloth=self.measure.nodes(),
            cloth_velocity=self.measure.cloth.get_simulation_nodal_velocities().cpu().numpy(),
            hardware=self.measure.bodies.get_transforms().cpu().numpy(),
            hardware_velocity=self.measure.bodies.get_velocities().cpu().numpy(),
            left_joints=self.left.data.joint_pos.torch.cpu().numpy(),right_joints=self.right.data.joint_pos.torch.cpu().numpy(),
            left_joint_velocities=self.left.data.joint_vel.torch.cpu().numpy(),
            right_joint_velocities=self.right.data.joint_vel.torch.cpu().numpy())
        self.trace=(destination/'trace.jsonl').open('w',buffering=1)
        if args.record:
            import imageio.v2 as imageio
            for name in CAMERAS:
                self.writers[name]=imageio.get_writer(str(destination/(name+'.mp4')),fps=self.hz,
                    codec='libx264',pixelformat='yuv420p',macro_block_size=1)
            # Rendering converges the initial images without advancing physics.
            before=self.measure.index
            nodes_before=self.measure.nodes()
            hardware_before=self.measure.bodies.get_transforms().cpu().numpy().copy()
            for _ in range(4):base.sim.render()
            assert self.measure.index==before, 'Rendering advanced native physics'
            assert np.array_equal(nodes_before,self.measure.nodes()), 'Rendering changed fabric state'
            assert np.array_equal(hardware_before,self.measure.bodies.get_transforms().cpu().numpy()), 'Rendering changed hardware state'
            write(destination/'render-observation-audit.json',dict(render_calls=4,physics_steps_advanced=0,
                cloth_and_hardware_unchanged=True))

    @property
    def time_s(self):
        return len(self.rows)/self.hz

    def joint_positions(self):
        return self.base.observation_manager.compute()['policy']['joint_pos'][0].cpu().numpy().copy()

    def images(self):
        assert self.args.record, 'Policy observation requires all three cameras'
        images={}
        for name in CAMERAS:
            pixels=self.base.scene[name].data.output['rgb']
            pixels=pixels.torch if hasattr(pixels,'torch') else pixels
            images[name]=pixels[0,...,:3].cpu().numpy().astype(np.uint8)
        return images

    def observe(self):
        """Return RGB, measured joints, instruction and simulation timestamps only."""
        index=self.measure.index
        result=observation(self.images(),self.joint_positions(),self.time_s,self.instruction)
        assert self.measure.index==index, 'Observation advanced native physics'
        return result

    def step(self, requested, control=None):
        """Apply exactly one standard action period and return its measured outcome."""
        from inspect_robots.types import StepResult
        assert not self.closed and self.reason is None, 'Episode has terminated'
        requested=validate_action(requested)
        tensor=self.torch.as_tensor(requested[None],device=self.base.device,dtype=self.torch.float32)
        self.env.step(tensor)
        executed=np.concatenate([self.base.action_manager.get_term(name).processed_actions[0].cpu().numpy()
            for name in self.base.action_manager.active_terms])
        step=len(self.rows)
        row=jsonable(dict(step=step,time_s=(step+1)/self.hz,requested_action=requested,robot_action=executed,
            joint_positions=self.joint_positions(),measured_left_tcp_m=self.tcp_world(self.left),
            measured_right_tcp_m=self.tcp_world(self.right),
            left_link_poses_xyzw=self.left.data.body_link_pose_w.torch[0].cpu().numpy(),left_link_names=list(self.left.body_names),
            right_link_poses_xyzw=self.right.data.body_link_pose_w.torch[0].cpu().numpy(),right_link_names=list(self.right.body_names),
            control=control or {'phase':'external_action'},physics=self.measure.last,
            original_hanger_task_success=False))
        nodes=self.measure.nodes()
        row['robot_clearance_m']=robot_clearance(row,nodes,self.meta,self.meta['robot_collision_bounds'])
        score=self.scorer.observe_control(row)
        row.update(task_score=score['task_score'],online_prefix_score=score['task_score'],
                   full_task_success=score['full_task_success'])
        self.rows.append(row);self.nodes.append(nodes)
        self.trace.write(json.dumps(row,allow_nan=True)+'\n')
        images=self.images() if self.args.record else None
        for name,writer in self.writers.items():writer.append_data(images[name])
        if (getattr(self.measure,'pilot_peak_table_force',0.)>25.
                or getattr(self.measure,'pilot_peak_interarm_force',0.)>.5):
            self.reason='pilot_rigid_contact_guard'
        elif score['invalid_reasons']:
            self.reason='invalid_simulation_state'
        elif score['full_task_success']:
            self.reason='success'
        elif len(self.rows)>=min(max_actions(self.cfg),round(self.args.seconds*self.hz)):
            self.reason='time_limit'
        public=observation(images,self.joint_positions(),self.time_s,self.instruction) if images is not None else None
        if step%self.hz==0:
            print(json.dumps(jsonable(dict(step=step,time_s=self.time_s,score=score['task_score'],
                max_strain=self.measure.max_strain,control=control,termination_reason=self.reason))),flush=True)
        return StepResult(observation=public,reward=score['task_score'],
            terminated=self.reason not in (None,'time_limit'),truncated=self.reason=='time_limit',
            termination_reason=self.reason,info=score)

    def finish(self, reason=None):
        """Finalize evidence at the current state; never execute extra actions."""
        if self.closed:return self.result
        self.reason=self.reason or reason or 'closed'
        self.closed=True
        for writer in self.writers.values():writer.close()
        self.trace.close()
        self.measure.close()
        np.savez_compressed(self.destination/'control-steps.npz',
            actions=np.asarray([row['robot_action'] for row in self.rows]).reshape(-1,14),
            cloth_nodes=np.asarray(self.nodes).reshape(-1,self.measure.count,3))
        np.savez_compressed(self.destination/'worst-table-contact.npz',**self.measure.worst_table_contact)
        with (self.destination/'physics.jsonl').open() as stream:
            physics=[json.loads(line) for line in stream]
        result=task_score(self.rows,physics,self.cfg,self.meta,self.reason)
        scores=result.pop('per_control_scores')
        assert result['task_score']==self.scorer.finalize()['task_score'], 'Online/offline terminal score disagreement'
        if not result['invalid_reasons']:
            assert scores==[row['online_prefix_score'] for row in self.rows], 'Online/offline prefix disagreement'
        with (self.destination/'trace.jsonl').open('w') as trace,(self.destination/'scores.jsonl').open('w') as score_log:
            for row,score in zip(self.rows,scores,strict=True):
                row.update(task_score=score,full_task_success=score==1.)
                trace.write(json.dumps(row,allow_nan=True)+'\n')
                score_log.write(json.dumps(dict(step=row['step'],time_s=row['time_s'],task_score=score,
                    full_task_success=score==1.,invalid_reasons=result['invalid_reasons']))+'\n')
        result.update(case='fasten_three',seed=self.args.seed,bundle_sha256=self.bundle,
            control_steps=len(self.rows),physics_steps_observed=self.measure.index,
            runtime_seconds=time.monotonic()-self.started,control_hz=self.hz,
            termination_reason=self.reason,reset_mode='fresh_native_process')
        if getattr(self.args,'pilot_diagnostics',False):
            result['pilot_peak_table_contact_N']=self.measure.pilot_peak_table_force
            result['pilot_peak_interarm_contact_N']=self.measure.pilot_peak_interarm_force
            result['pilot_rigid_contact_guard_passed']=(self.measure.pilot_peak_table_force<=25.
                and self.measure.pilot_peak_interarm_force<=.5)
        write(self.destination/'result.json',result)
        self.result=result
        return result
