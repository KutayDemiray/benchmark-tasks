"""Native tabletop shirt diagnostics, using only stock ordinary robot actions."""

import argparse
import copy
import json
import os
from pathlib import Path
from task_paths import OUTPUT_ROOT
import time
import traceback

import numpy as np

from bench import snapshot,write,jsonable
from garment_runner import Measurements,verify_scene
from robot_runner import start_app,make_environment
from tabletop_scene import configure,build,camera_rig
from tabletop_scoring import support_score,press_score,task_score
from tabletop_clearance import robot_clearance,stock_collision_bounds

HERE=Path(__file__).resolve().parent


def table_observation(cfg):
    """Measure full-step cloth penetration of the physical tabletop plane."""
    p=cfg['tabletop']
    def observe(measure,nodes,poses):
        torch=measure.torch
        center=nodes.new_tensor(p['table_center_xy_m'])
        half=nodes.new_tensor(p['table_size_xy_m'])/2
        inside=(torch.abs(nodes[:,:2]-center)<half).all(dim=1)
        gaps=torch.where(inside,nodes[:,2]-p['table_top_m'],float('inf'))
        gap,index=gaps.min(dim=0)
        values=torch.stack((torch.clamp(-gap,min=0),inside.double().mean(),index.double(),*nodes[index])).cpu().numpy()
        result=dict(table_penetration_m=float(values[0]),table_coverage_fraction=float(values[1]),
            lowest_table_node_index=int(values[2]),lowest_table_node_position_m=values[3:6].tolist())
        if float(values[0])>getattr(measure,'max_table_penetration',-1.):
            measure.worst_table_contact=dict(physics_step=measure.index,nodes=nodes.cpu().numpy(),
                hardware=poses.cpu().numpy(),penetration_m=float(values[0]))
        measure.max_table_penetration=max(getattr(measure,'max_table_penetration',0.),float(values[0]))
        if hasattr(measure,'robot_contacts'):
            force=measure.robot_contacts.get_contact_force_matrix(measure.base.physics_dt)
            summed=force.sum(dim=1)
            magnitude=torch.linalg.vector_norm(force,dim=-1).sum(dim=1)
            contact=torch.cat((summed,magnitude[:,None]),dim=1).cpu().numpy()
            measure.max_robot_force=max(measure.max_robot_force,float(magnitude.max().item()))
            result['snap_robot_contact_forces_N']={k:contact[i,:3].tolist() for k,i in measure.contact_indices.items()}
            result['snap_robot_contact_norm_N']={k:float(contact[i,3]) for k,i in measure.contact_indices.items()}
        return result
    return observe


def add_contact_measurements(measure,meta,dest):
    """Observe robot/hardware contacts; no forces or attachments are commanded."""
    from yam_arena.robot import asset_link_paths
    paths=asset_link_paths();env_root=meta['root'].rsplit('/',1)[0]
    filters=[env_root+'/'+arm+'/'+suffix.lstrip('/') for arm in ('LeftArm','RightArm') for suffix in paths.values()]
    names=[name for name in meta['hardware'] if name.startswith(('male','socket')) and '_ball' not in name]
    measure.robot_contacts=measure.sim.create_rigid_contact_view(
        [meta['hardware'][n] for n in names],filter_patterns=[filters]*len(names))
    assert measure.robot_contacts.sensor_count==len(names) and measure.robot_contacts.filter_count==len(filters)
    measure.contact_indices={n:list(measure.robot_contacts.sensor_paths).index(meta['hardware'][n]) for n in names}
    measure.max_robot_force=0.
    if measure.strain_witness is not None:
        measure.robot_bodies=measure.sim.create_rigid_body_view(filters)
        write(dest/'strain-robot-views.json',dict(paths=measure.robot_bodies.prim_paths))
    write(dest/'contact-sensors.json',dict(sensor_paths=measure.robot_contacts.sensor_paths,robot_filters=filters,
        contact_indices=measure.contact_indices,scope='Rigid normal contact only; does not measure distributed robot/cloth forces.'))


def run_episode(env,cfg,args,meta,dest,bundle):
    import torch
    from yam_arena.runtime import tcp_world
    env.reset()
    base=env.unwrapped
    assert base.action_manager.total_action_dim==14 and abs(base.step_dt-.1)<1e-9
    assert list(base.action_manager.active_terms)==['left_arm','left_gripper','right_arm','right_gripper']
    verify_scene(base,meta,dest,args.record)
    meta['robot_collision_bounds']=stock_collision_bounds()
    write(dest/'setup.json',dict(parameters=cfg,seed=args.seed,case=args.case,stage='tabletop_shirt',
        recorded_cameras=args.record,control_hz=10.,action_dimension=14,bundle_sha256=bundle))
    write(dest/'topology.json',meta)
    measure=Measurements(base,meta,dest,round(args.contact_substeps_seconds*cfg['simulation']['physics_hz']),
                         extra_observer=table_observation(cfg),strain_witness=args.strain_witness)
    add_contact_measurements(measure,meta,dest)
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
    parked=cfg['garment']['parked_robot_joints_rad']
    action=np.concatenate((parked['left'],[1.],parked['right'],[1.]))
    controller=None
    if args.case in ('press_one','fasten_three'):
        from tabletop_control import TablePressController
        from yam_arena.runtime import arm_q
        targets=(0,) if args.case=='press_one' else (0,1,2)
        controller=TablePressController(arm_q(left),tcp_world(left),cfg,targets=targets,
            observer=args.case=='fasten_three')
    try:
        with (dest/'trace.jsonl').open('w',buffering=1) as trace:
            for step in range(round(args.seconds*10)):
                info=dict(phase='passive_settle')
                if controller is not None:
                    action,info=controller.action(step*.1,tcp_world(left),measure.last)
                env.step(torch.as_tensor(action[None],device=base.device,dtype=torch.float32))
                row=jsonable(dict(step=step,time_s=(step+1)*.1,robot_action=action,
                    joint_positions=base.observation_manager.compute()['policy']['joint_pos'][0].cpu().numpy(),
                    measured_left_tcp_m=tcp_world(left),measured_right_tcp_m=tcp_world(right),
                    left_link_poses_xyzw=left.data.body_link_pose_w.torch[0].cpu().numpy(),left_link_names=list(left.body_names),
                    right_link_poses_xyzw=right.data.body_link_pose_w.torch[0].cpu().numpy(),right_link_names=list(right.body_names),
                    control=info,physics=measure.last,task_score=0.,full_task_success=False,
                    original_hanger_task_success=False))
                cloth_nodes=measure.nodes()
                row['robot_clearance_m']=robot_clearance(row,cloth_nodes,meta,meta['robot_collision_bounds'])
                rows.append(row);trace.write(json.dumps(row,allow_nan=True)+'\n');nodes.append(cloth_nodes)
                for name,writer in writers.items():
                    pixels=base.scene[name].data.output['rgb']
                    pixels=pixels.torch if hasattr(pixels,'torch') else pixels
                    writer.append_data(pixels[0,...,:3].cpu().numpy().astype(np.uint8))
                if step%10==0:
                    print(json.dumps(dict(step=step,max_strain=measure.max_strain,max_rivet_error_m=measure.max_rivet,
                        table_penetration_m=measure.last['table_penetration_m'],control=jsonable(info))),flush=True)
                if not measure.finite or measure.max_rivet>cfg['gate']['max_attachment_error_m']:
                    reason='invalid_fabric_state';break
                if measure.max_strain>cfg['gate']['max_edge_strain']:
                    reason='invalid_fabric_state';break
                if measure.max_table_penetration>cfg['tabletop']['max_table_penetration_m']:
                    reason='invalid_table_contact';break
                if getattr(measure,'max_robot_force',0.)>25.:
                    reason='invalid_robot_force';break
                if controller is not None and info['completed']:
                    reason='controller_aborted' if info['abort_reason'] else 'controller_completed';break
    finally:
        for writer in writers.values():writer.close()
        measure.close()
    np.savez_compressed(dest/'control-steps.npz',actions=np.array([r['robot_action'] for r in rows]),cloth_nodes=np.array(nodes))
    np.savez_compressed(dest/'worst-table-contact.npz',**measure.worst_table_contact)
    physical=[json.loads(s) for s in (dest/'physics.jsonl').read_text().splitlines()]
    result=dict(case=args.case,seed=args.seed,bundle_sha256=bundle,termination_reason=reason,
        runtime_seconds=time.monotonic()-start,physics_steps_observed=measure.index,control_steps=len(rows))
    scorer={'settle':support_score,'press_one':press_score,'fasten_three':task_score}[args.case]
    result.update(scorer(rows,physical,cfg,meta,reason))
    if args.case=='fasten_three':
        scores=result.pop('per_control_scores')
        with (dest/'scores.jsonl').open('w') as stream:
            for row,score in zip(rows,scores,strict=True):
                row.update(task_score=score,full_task_success=score==1.)
                stream.write(json.dumps(dict(step=row['step'],time_s=row['time_s'],task_score=score,
                    full_task_success=score==1.,invalid_reasons=result['invalid_reasons']))+'\n')
        with (dest/'trace.jsonl').open('w') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=True)+'\n')
    write(dest/'result.json',result);print('RESULT '+json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',choices=('settle','press_one','fasten_three'),default='settle')
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--reset-check',action='store_true',
        help='After the requested episodes, rebuild and observe the open shirt for five seconds in the same process.')
    parser.add_argument('--strain-witness',action='store_true',help='Capture exact peak strain and eight full native states on each side.')
    parser.add_argument('--cloth-solver-iterations',type=int,choices=(64,128),default=64)
    parser.add_argument('--body-mesh-pitch',type=float,choices=(.01,.005),default=.01)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--physics-hz',type=int,choices=(960,1920),default=960,
        help='Numerical timestep convergence comparison; never changes the 10 Hz action rate or physical parameters.')
    parser.add_argument('--cloth-contact-offset',type=float,default=.0003,
        help='Numerical contact-generation distance in metres; physical rest offset remains 0.15 mm.')
    parser.add_argument('--contact-substeps-seconds',type=float,default=.5)
    args=parser.parse_args()
    assert 0<args.seconds<=300 and args.episodes>=1
    assert 0<=args.contact_substeps_seconds<=min(args.seconds,1.)
    from gpu_runtime import inherited_fds
    inherited_fds()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    cfg=json.loads((HERE/'parameters.json').read_text());configure(cfg)
    cfg['simulation']['physics_hz']=args.physics_hz
    cfg['fabric']['solver_iterations']=args.cloth_solver_iterations
    cfg['garment']['body_mesh_pitch_m']=args.body_mesh_pitch
    assert cfg['fabric']['rest_offset_m']<args.cloth_contact_offset<=.002
    cfg['fabric']['contact_offset_m']=args.cloth_contact_offset
    cfg['tabletop_test']=dict(case=args.case,contact_substeps_seconds=args.contact_substeps_seconds,
        right_camera_observer=args.case=='fasten_three')
    cfg['episode_seed']=args.seed
    bundle=snapshot(args.out);app=start_app(args.out,args.record);env=None
    try:
        import omni.usd
        results=[]
        for index in range(args.episodes+int(args.reset_check)):
            episode_args=copy.copy(args)
            if index==args.episodes:
                episode_args.case='settle';episode_args.seconds=5.
            cfg['tabletop_test'].update(case=episode_args.case,right_camera_observer=episode_args.case=='fasten_three')
            omni.usd.get_context().new_stage();meta={}
            dest=args.out/f'episode-{index:02d}';dest.mkdir()
            env=make_environment(cfg,args.record,meta,args.out,scene_builder=build,rig=camera_rig())
            results.append(run_episode(env,cfg,episode_args,meta,dest,bundle))
            env.close();env=None
        write(args.out/'results.json',results)
    except BaseException as exc:
        write(args.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),full_task_success=False))
        traceback.print_exc();raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':
    main()
