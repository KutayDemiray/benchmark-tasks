"""Native ordinary-action garment diagnostics; no fixture control or grasp locks."""

from task_paths import OUTPUT_ROOT

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np

from bench import snapshot,write,jsonable
from robot_runner import start_app,make_environment

HERE=Path(__file__).resolve().parent


class Measurements:
    """Read every native substep; evaluate cloth integrity on the physics GPU."""

    def __init__(self,base,meta,dest,contact_substeps=0,extra_observer=None,strain_witness=False):
        import torch
        import omni.physics.tensors as tensors
        self.torch=torch
        self.sim=tensors.create_simulation_view('torch')
        self.sim.set_subspace_roots('/')
        assert self.sim.device_ordinal==0
        self.bodies=self.sim.create_rigid_body_view(list(meta['hardware'].values()))
        self.body_indices={k:list(self.bodies.prim_paths).index(p) for k,p in meta['hardware'].items()}
        assert self.bodies.count==len(meta['hardware'])
        self.cloth=self.sim.create_surface_deformable_body_view(meta['cloth']['path'])
        d=meta['cloth']
        self.count=len(d['initial'])
        self.edges=torch.as_tensor(d['edges'],device=base.device,dtype=torch.long)
        self.lengths=torch.as_tensor(d['rest_lengths'],device=base.device,dtype=torch.float64)
        ids,local,body_index=[],[],[]
        for att in d['attachments']:
            ids.extend(att['ids'])
            local.extend(att['local'])
            body_index.extend([self.body_indices[att['hardware']]]*len(att['ids']))
        self.rivet_ids=torch.as_tensor(np.array(ids),device=base.device,dtype=torch.long)
        self.local=torch.as_tensor(np.array(local).reshape(-1,3),device=base.device,dtype=torch.float64)
        self.rivet_bodies=torch.as_tensor(np.array(body_index),device=base.device,dtype=torch.long)
        self.index=0
        self.max_strain=0.
        self.max_rivet=0.
        self.finite=True
        self.last=None
        self.meta=meta
        self.extra_observer=extra_observer
        self.substep_nodes=None
        self.strain_witness=None
        self.destination=dest
        if strain_witness:
            from strain_witness import StrainWitness
            self.strain_witness=StrainWitness(d['edges'],d['rest_lengths'],d['faces'])
        if contact_substeps:
            # A short, opt-in geometry diagnostic resolves crossings that a
            # 10 Hz video/state recording can miss. It never changes physics.
            self.substep_nodes=np.lib.format.open_memmap(dest/'contact-substeps.npy',
                mode='w+',dtype=np.float32,shape=(contact_substeps+1,self.count,3))
        self.log=(dest/'physics.jsonl').open('w',buffering=1)
        self.original_step=base.sim.step
        self.base=base
        def measured_step(*args,**kwargs):
            self.original_step(*args,**kwargs)
            self.observe()
        base.sim.step=measured_step
        write(dest/'measurement-views.json',dict(body_indices=self.body_indices,
              cloth_prim_paths=self.cloth.prim_paths,nodes=self.count,physics_device_ordinal=self.sim.device_ordinal))

    def nodes(self):
        return self.cloth.get_simulation_nodal_positions()[0,:self.count].cpu().numpy().copy()

    def observe(self):
        torch=self.torch
        nodes=self.cloth.get_simulation_nodal_positions()[0,:self.count].double()
        poses=self.bodies.get_transforms().double()
        lengths=torch.linalg.vector_norm(nodes[self.edges[:,0]]-nodes[self.edges[:,1]],dim=-1)
        strains=torch.abs(lengths/self.lengths-1)
        strain,edge_index=torch.max(strains,dim=0)
        rivet_poses=poses[self.rivet_bodies]
        q,w=rivet_poses[:,3:6],rivet_poses[:,6:7]
        rotated=self.local+2*torch.linalg.cross(q,torch.linalg.cross(q,self.local)+w*self.local)
        error=(torch.linalg.vector_norm(nodes[self.rivet_ids]-(rotated+rivet_poses[:,:3]),dim=-1).max()
               if len(self.rivet_ids) else strain.new_tensor(0.))
        finite=torch.isfinite(nodes).all() & torch.isfinite(poses).all()
        summaries=torch.stack((strain,error,finite.double(),nodes[:,2].min(),nodes[:,2].max(),*nodes.mean(dim=0),edge_index.double())).cpu().numpy()
        body_array=poses.cpu().numpy()
        finite=bool(summaries[2])
        self.finite &= finite
        if finite:
            self.max_strain=max(self.max_strain,float(summaries[0]))
            self.max_rivet=max(self.max_rivet,float(summaries[1]))
        self.last=dict(physics_step=self.index,hardware={k:body_array[i].tolist() for k,i in self.body_indices.items()},
                       strain=float(summaries[0]),rivet_error_m=float(summaries[1]),finite=finite,
                       min_cloth_z_m=float(summaries[3]),max_cloth_z_m=float(summaries[4]),
                       mean_cloth_position_m=summaries[5:8].tolist(),worst_strain_edge_index=int(summaries[8]))
        if self.extra_observer is not None:
            self.last.update(self.extra_observer(self,nodes,poses))
        if self.substep_nodes is not None and self.index<len(self.substep_nodes):
            self.substep_nodes[self.index]=nodes.float().cpu().numpy()
        if self.strain_witness is not None:
            state=dict(physics_step=self.index,reported_strain=float(summaries[0]),
                reported_edge_index=int(summaries[8]),cloth_positions_m=nodes.float().cpu().numpy().copy(),
                cloth_velocities_m_s=self.cloth.get_simulation_nodal_velocities()[0,:self.count].cpu().numpy().copy(),
                hardware_poses_xyzw=body_array.copy(),hardware_velocities=self.bodies.get_velocities().cpu().numpy().copy())
            if hasattr(self,'robot_bodies'):
                state['robot_poses_xyzw']=self.robot_bodies.get_transforms().cpu().numpy().copy()
                state['robot_velocities']=self.robot_bodies.get_velocities().cpu().numpy().copy()
            self.strain_witness.record(state)
        if hasattr(self,'state_observer'):
            self.state_observer(self.last)
        # A nonfinite native state remains an explicit invalid record, not a success.
        self.log.write(json.dumps(self.last,allow_nan=True)+'\n')
        self.index+=1

    def close(self):
        self.base.sim.step=self.original_step
        self.log.close()
        if self.substep_nodes is not None:
            self.substep_nodes.flush()
        if self.strain_witness is not None:
            self.strain_witness.save(self.destination)
        self.sim.invalidate()


def verify_world_joint(path,body0,body1,meta):
    """Permit only the explicitly declared diagnostic hanger mount."""
    if body0 and body1:
        return False
    assert meta.get('diagnostic_world_mounts',False),'World-mounted task body'
    assert path==meta.get('allowed_diagnostic_world_joint'),'Undeclared world joint'
    assert not body0 and body1==[meta['hardware']['hanger']],'Wrong diagnostic mount body'
    return True


def verify_scene(base,meta,dest,record):
    """Verify both GPU assignments and the absence of hidden scene attachments."""
    import carb
    import omni.usd
    from pxr import UsdPhysics
    from gpu_runtime import verify_environment
    verify_environment(base,meta,dest,record)
    stage=omni.usd.get_context().get_stage()
    attachments=[]
    world_mounts=[]
    allowed={meta['hardware'][a['hardware']] for a in meta['cloth']['attachments']}
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(meta['root']):
            continue
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            path=str(prim.GetPath())
            if verify_world_joint(path,[str(p) for p in joint.GetBody0Rel().GetTargets()],
                                  [str(p) for p in joint.GetBody1Rel().GetTargets()],meta):
                world_mounts.append(path)
        if prim.GetTypeName()=='OmniPhysicsVtxXformAttachment':
            src0=[str(p) for p in prim.GetRelationship('omniphysics:src0').GetTargets()]
            src1=[str(p) for p in prim.GetRelationship('omniphysics:src1').GetTargets()]
            assert src0==[meta['cloth']['path']] and len(src1)==1 and src1[0] in allowed
            attachments.append(dict(path=str(prim.GetPath()),target=src1))
    assert len(attachments)==len(meta['cloth']['attachments'])
    assert len(world_mounts)==int(meta.get('diagnostic_world_mounts',False))
    write(dest/'attachment-audit.json',dict(local_rivets=attachments,world_mounts=bool(world_mounts),
          declared_diagnostic_world_joints=world_mounts,scope='diagnostic_only',
          robot_grasp_attachments=False,shirt_to_hanger_attachments=False,hanger_to_rail_joint=False))
    stage.GetRootLayer().Export(str(dest/'initial.usda'))


def passive_score(rows,cfg,reason,max_strain,max_rivet,finite):
    """Score only an explicitly intermediate settling diagnostic."""
    from robot_geometry import rotation
    from robot_scoring import snap_seated
    finite=bool(finite)
    intact=finite and max_strain<=cfg['gate']['max_edge_strain'] and max_rivet<=cfg['gate']['max_attachment_error_m']
    tail=rows[-20:]
    centers=np.array([r['physics']['mean_cloth_position_m'] for r in tail])
    drift=float(np.linalg.norm(centers-centers[-1],axis=1).max()) if len(centers) else None
    seated=[]
    for row in rows:
        for index in range(len(cfg['garment']['snap_heights_from_hem_m'])):
            poses=row['physics']['hardware']
            m,s=np.array(poses[f'male{index}']),np.array(poses[f'socket{index}'])
            d=rotation(s[3:]).T@(m[:3]-s[:3])
            if snap_seated(m,s,cfg):
                seated.append(dict(step=row['step'],pair=index,axial_m=float(d[2])))
    cloth_above_floor=bool(tail and min(r['physics']['min_cloth_z_m'] for r in tail)>.10)
    hanger_on_rail=bool(tail and all(abs(r['physics']['hardware']['hanger'][2]-cfg['hanger']['origin_m'][2])<.04 for r in tail))
    checks=dict(intact=intact,cloth_above_floor=cloth_above_floor,hanger_on_rail=hanger_on_rail,
                settling_drift_under_2cm=bool(drift is not None and drift<.02),no_self_closing=not seated,
                observation_completed=reason=='diagnostic_completed' and len(rows)>=50)
    case=cfg.get('garment_test',{}).get('case','settle')
    if case=='closed_target':
        checks.pop('no_self_closing')
        checks['preclosed_target_retained']=bool(tail and all(snap_seated(
            r['physics']['hardware']['male0'],r['physics']['hardware']['socket0'],cfg) for r in rows))
        checks['other_pairs_stay_open']=not any(e['pair']!=0 for e in seated)
    if case=='loaded_sleeve_probe':
        checks['payload_supported']=bool(tail and all(
            r['physics']['hardware']['payload'][2]>cfg['hanger']['origin_m'][2]-.22 for r in tail))
    notes={
        'settle':'Passive shirt support diagnostic; no robot fastening or table-to-hanger task demonstrated.',
        'closed_target':'Upper snap starts closed for a support diagnostic; no robot fastening or table-to-hanger task demonstrated.',
        'support_probe':'Small flat fabric patch contact control; no shirt or task demonstrated.',
        'sleeve_probe':'Small closed fabric sleeve contact control; no shirt or task demonstrated.',
        'elongated_sleeve_probe':'Unloaded elongated fabric-loop contact control; no shirt or task demonstrated.',
        'loaded_sleeve_probe':'Fabric loop supporting a free steel test rod; load/contact control only, no shirt or task demonstrated.',
    }
    fixed=cfg.get('garment_test',{}).get('fixed_support',False)
    note=notes[case]
    if fixed:
        note+=' Hanger is world-fixed for this comparison; this cannot validate free-hanger support.'
    return dict(diagnostic_pass=all(checks.values()),checks=checks,final_two_seconds_mean_node_drift_m=drift,
                self_seated_events=seated,full_task_score=0.,full_task_success=False,
                free_hanger_support_validated=False if fixed else None,note=note)


def run_episode(env,cfg,args,meta,dest,bundle):
    import torch
    from yam_arena.constants import READY
    from yam_arena.runtime import tcp_world
    env.reset()
    base=env.unwrapped
    assert base.action_manager.total_action_dim==14
    assert list(base.action_manager.active_terms)==['left_arm','left_gripper','right_arm','right_gripper']
    assert abs(base.step_dt-.1)<1e-9
    verify_scene(base,meta,dest,args.record)
    write(dest/'setup.json',dict(parameters=cfg,seed=args.seed,case=args.case,stage='garment_diagnostic',
          recorded_cameras=args.record,control_hz=10.,action_dimension=14,bundle_sha256=bundle))
    write(dest/'topology.json',meta)
    measure=Measurements(base,meta,dest,round(args.contact_substeps_seconds*cfg['simulation']['physics_hz']))
    measure.observe()
    left,right=base.scene['left_robot'],base.scene['right_robot']
    np.savez_compressed(dest/'initial-measured-state.npz',cloth=measure.nodes(),
        cloth_velocity=measure.cloth.get_simulation_nodal_velocities().cpu().numpy(),
        hardware=measure.bodies.get_transforms().cpu().numpy(),hardware_velocity=measure.bodies.get_velocities().cpu().numpy(),
        left_joints=left.data.joint_pos.torch.cpu().numpy(),right_joints=right.data.joint_pos.torch.cpu().numpy(),
        left_joint_velocities=left.data.joint_vel.torch.cpu().numpy(),right_joint_velocities=right.data.joint_vel.torch.cpu().numpy())
    writers={}
    if args.record:
        import imageio.v2 as imageio
        for name in ('top_cam','left_cam','right_cam'):
            writers[name]=imageio.get_writer(str(dest/(name+'.mp4')),fps=10,codec='libx264',pixelformat='yuv420p',macro_block_size=1)
    rows,nodes=[],[]
    reason='diagnostic_completed'
    start=time.monotonic()
    parked=cfg['garment'].get('parked_robot_joints_rad',dict(left=READY,right=READY))
    action=np.concatenate((parked['left'],[1.],parked['right'],[1.]))
    try:
        with (dest/'trace.jsonl').open('w',buffering=1) as trace:
            for step in range(round(args.seconds*10)):
                env.step(torch.as_tensor(action[None],device=base.device,dtype=torch.float32))
                row=jsonable(dict(step=step,time_s=(step+1)*.1,robot_action=action,
                    joint_positions=base.observation_manager.compute()['policy']['joint_pos'][0].cpu().numpy(),
                    measured_left_tcp_m=tcp_world(left),measured_right_tcp_m=tcp_world(right),
                    left_link_poses_xyzw=left.data.body_link_pose_w.torch[0].cpu().numpy(),left_link_names=list(left.body_names),
                    right_link_poses_xyzw=right.data.body_link_pose_w.torch[0].cpu().numpy(),right_link_names=list(right.body_names),
                    control=dict(phase='passive_settle'),physics=measure.last,full_task_score=0.,full_task_success=False))
                rows.append(row)
                trace.write(json.dumps(row,allow_nan=True)+'\n')
                nodes.append(measure.nodes())
                for name,writer in writers.items():
                    pixels=base.scene[name].data.output['rgb']
                    pixels=pixels.torch if hasattr(pixels,'torch') else pixels
                    writer.append_data(pixels[0,...,:3].cpu().numpy().astype(np.uint8))
                if step%10==0:
                    print(json.dumps(dict(step=step,max_strain=measure.max_strain,max_rivet_error_m=measure.max_rivet,
                        min_cloth_z_m=measure.last['min_cloth_z_m'],hanger_z=measure.last['hardware']['hanger'][2])),flush=True)
                if not measure.finite or measure.max_rivet>cfg['gate']['max_attachment_error_m']:
                    reason='invalid_fabric_state'
                    break
                if measure.max_strain>cfg['gate']['max_edge_strain']:
                    reason='invalid_fabric_state'
                    if not args.observe_invalid:
                        break
    finally:
        for writer in writers.values():
            writer.close()
        measure.close()
    np.savez_compressed(dest/'control-steps.npz',actions=np.array([r['robot_action'] for r in rows]),cloth_nodes=np.array(nodes))
    result=dict(stage='garment_passive_diagnostic',case=args.case,seed=args.seed,bundle_sha256=bundle,
                termination_reason=reason,runtime_seconds=time.monotonic()-start,physics_steps_observed=measure.index,
                control_steps=len(rows),max_cloth_strain=measure.max_strain,max_rivet_error_m=measure.max_rivet)
    result.update(passive_score(rows,cfg,reason,measure.max_strain,measure.max_rivet,measure.finite))
    write(dest/'result.json',result)
    print('RESULT '+json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',choices=('settle','support_probe','sleeve_probe','elongated_sleeve_probe','loaded_sleeve_probe','closed_target'),default='settle')
    parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--physics-hz',type=int,choices=(960,1920),help='Numerical time-step refinement; native control remains 10 Hz')
    parser.add_argument('--cloth-iterations',type=int,choices=(64,128),help='Numerical surface solver refinement; no material-law change')
    parser.add_argument('--observe-invalid',action='store_true',help='Diagnostic only: keep observing after strain invalidity; the run remains invalid and earns no credit')
    parser.add_argument('--contact-substeps-seconds',type=float,default=0.,help='Record full cloth positions at physics rate for an initial interval up to 1 second; diagnostic only')
    parser.add_argument('--hanger-contact-offset',type=float,help='Diagnostic contact-generation margin for hanger/rail only; geometry, rest offset and snap contacts stay fixed')
    parser.add_argument('--initial-height-offset',type=float,default=0.,help='Initial shirt vertical placement only; no runtime pose writes')
    parser.add_argument('--cloth-contact-updates',type=int,choices=(0,4,64),help='Numerical collision-pair refreshes per physics step; 0 is native adaptive mode')
    parser.add_argument('--fixed-support',action='store_true',help='Declared world-fixed hanger for the loaded-loop comparison ONLY; never task credit')
    parser.add_argument('--gentle-placement',action='store_true',help='Start the free hook 0.25 mm clear of a same-radius capsule rail; reset placement only')
    parser.add_argument('--wooden-hanger',action='store_true',help='Representative 430 mm x 14 mm wooden hanger; requires gentle-placement, diagnostic only')
    args=parser.parse_args()
    assert 0<args.seconds<=300 and args.episodes>=1
    assert 0<=args.contact_substeps_seconds<=min(args.seconds,1.)
    assert not args.fixed_support or args.case=='loaded_sleeve_probe'
    assert not args.gentle_placement or (not args.fixed_support and args.case in ('settle','closed_target'))
    assert not args.wooden_hanger or args.gentle_placement
    from gpu_runtime import inherited_fds
    inherited_fds()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    cfg=json.loads((HERE/'parameters.json').read_text())
    if args.physics_hz:
        cfg['simulation']['physics_hz']=args.physics_hz
    if args.cloth_iterations:
        cfg['fabric']['solver_iterations']=args.cloth_iterations
    if args.cloth_contact_updates is not None:
        cfg['garment']['collision_pair_updates']=args.cloth_contact_updates
    if args.hanger_contact_offset is not None:
        assert .0001<=args.hanger_contact_offset<=.003
        cfg['hanger']['contact_offset_m']=args.hanger_contact_offset
    assert -.06<=args.initial_height_offset<=.05
    if args.initial_height_offset:
        assert args.case in ('settle','closed_target')
        cfg['garment']['hanging_origin_m'][2]+=args.initial_height_offset
    cfg['garment_test']={'case':args.case,'observe_invalid':args.observe_invalid,
                         'fixed_support':args.fixed_support,
                         'gentle_placement':args.gentle_placement,
                         'contact_substeps_seconds':args.contact_substeps_seconds,
                         'initial_height_offset_m':args.initial_height_offset}
    if args.wooden_hanger:
        from hanger_shapes import configure_wooden_hanger
        configure_wooden_hanger(cfg)
    if args.gentle_placement:
        from garment_contact_audit import initial_hook_rail_gap
        h=cfg['hanger']
        original_z=h['origin_m'][2]
        lo,hi=original_z-.004,original_z
        for _ in range(40):
            h['origin_m'][2]=(lo+hi)/2
            if initial_hook_rail_gap(h)>.00025:
                hi=h['origin_m'][2]
            else:
                lo=h['origin_m'][2]
        h['origin_m'][2]=hi
        cfg['garment']['hanging_origin_m'][2]+=hi-original_z
        h['rail_geometry']='capsule'
        cfg['garment_test']['gentle_placement_record']=dict(
            hook_rail_clearance_bound_m=initial_hook_rail_gap(h),hanger_height_shift_m=hi-original_z,
            rail_representation='Same 8 mm radius / 2 m shaft; capsule endcaps extend 8 mm beyond remote posts.',
            reason='Archived settled hook pose overlaps authored rail. Use geometric separation, not that invalid measured pose.')
    if args.case in ('support_probe','sleeve_probe','elongated_sleeve_probe','loaded_sleeve_probe'):
        cfg['garment']['snap_heights_from_hem_m']=[]
    bundle=snapshot(args.out)
    app=start_app(args.out,args.record)
    env=None
    try:
        import omni.usd
        from garment_scene import build,camera_rig
        results=[]
        for index in range(args.episodes):
            omni.usd.get_context().new_stage()
            meta={}
            dest=args.out/f'episode-{index:02d}'
            dest.mkdir()
            cfg['episode_seed']=args.seed
            env=make_environment(cfg,args.record,meta,args.out,scene_builder=build,rig=camera_rig())
            results.append(run_episode(env,cfg,args,meta,dest,bundle))
            env.close()
            env=None
        write(args.out/'results.json',results)
    except BaseException as exc:
        write(args.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),full_task_success=False))
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__=='__main__':
    main()
