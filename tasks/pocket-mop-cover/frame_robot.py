"""Stock-robot frame handling and optional qualified two-pocket integration."""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gate import array,solve_pose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    request=p.parse_args().request
    req=json.loads(request.read_text());out=request.parent
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    app=env=session=None
    report=dict(status='initializing',diagnostic_only=bool(req.get('release_probe_run')) or not req.get('robot_cover',False),
        full_task_success=False,score_status='Pending independent native outcome audit')
    if req.get('robot_cover',False):
        report['isolated_gates_prevalidated_by_launcher']=True
    try:
        from isaaclab.app import AppLauncher
        kit = kit_arguments(out)
        app=AppLauncher(headless=True,enable_cameras=True,device='cuda:0',multi_gpu=False,kit_args=kit).app
        import omni.usd
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from yam_arena import station
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback,task_room
        from frame_asset import frame_asset
        from robot_session import RobotSession
        from robot_contacts import collect
        from runtime_checks import verify_gpu
        from station_coordinates import TRANSLATION_S_W,translated_rig,translated_room,verify_station
        from cloth_solver import install as install_cloth_solver
        install_cloth_solver(req,out)
        from frame_solver import install as install_frame_solver
        install_frame_solver(req,out)
        if req.get('release_probe_run'):
            from release_physics import install
            install(req,out)
        shift=TRANSLATION_S_W.copy() if req.get('table_origin',False) else np.zeros(3)
        table_top=station.TABLE_TOP+shift[2]
        origin=np.array([.35,-.08,station.TABLE_TOP+(.177 if req.get('frame_press',False) else .004)])+shift
        metadata={}
        cover=None
        if req.get('robot_cover',False) and not req.get('release_probe_without_cover',False):
            from full_cover import FullCover
            cover=FullCover(req,out,np.array([origin[0]+req.get('cover_offset_x',0.),origin[1]+.14,table_top]))
        stock_rig=station.rig()
        embodiment=BimanualYamEmbodiment(enable_cameras=True,rig=translated_rig(stock_rig,shift),
            cam_width=req.get('camera_width',640),cam_height=req.get('camera_height',360))
        base_configure=env_cfg_callback(20)
        def configure(cfg):
            cfg=base_configure(cfg)
            cfg.sim.dt=1/req['physics_hz'];cfg.decimation=req['physics_hz']//20
            cfg.sim.render_interval=cfg.decimation;cfg.sim.use_fabric=False;cfg.sim.log_dir=str(out)
            if req.get('release_probe_run'):
                from release_physics import configure_scene
                configure_scene(req,cfg)
            return cfg
        assets=[*translated_room(stock_rig,shift),frame_asset(origin,metadata,
                snap_throat=req.get("snap_throat",False),folded=req.get('frame_press',False),fold_angle_deg=65.)]
        if cover is not None:assets.append(cover.asset)
        definition=IsaacLabArenaEnvironment(name='pocket_mop_frame_handling',
            scene=Scene(assets),
            embodiment=embodiment,env_cfg_callback=configure)
        env=ArenaEnvBuilder(definition,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cuda:0')).make_registered()
        base=env.unwrapped
        env.reset(seed=req['seed'])
        frame=base.scene['frame']
        if cover is not None:cover.bind(base)
        (out/'frame-dynamics.json').write_text(json.dumps(dict(
            body_names=list(frame.body_names),joint_names=list(frame.joint_names),
            masses_kg=array(frame.root_view.get_masses()).tolist(),
            local_com_poses_xyzw=array(frame.root_view.get_coms()).tolist(),
            inertias_kg_m2=array(frame.root_view.get_inertias()).tolist(),
            joint_stiffnesses=array(frame.root_view.get_dof_stiffnesses()).tolist(),
            joint_dampings=array(frame.root_view.get_dof_dampings()).tolist(),
            scope='Read-only native mass, center-of-mass, inertia and drive properties.'),indent=2))
        verify_gpu(out,base.sim,frame)
        if req.get('reset_check',False):
            from episode_checks import verify_resets
            verify_resets(env,frame,cover,req,out)
        stage=omni.usd.get_context().get_stage()
        if req.get('release_probe_run'):
            from release_physics import verify_scene
            verify_scene(req,stage,base.cfg.sim.physics_prim_path,out)
        native_station=verify_station(stage,stock_rig,shift)
        (out/"station-pose-check.json").write_text(json.dumps(native_station,indent=2))
        from pxr import UsdPhysics
        constraints=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint) and 'Frame' in str(p.GetPath())]
        assert all('Laboratory' not in path for path in constraints)
        assert not any('Attachment' in p.GetTypeName() for p in stage.Traverse())
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        (out/'frame-geometry.json').write_text(json.dumps(metadata,indent=2))
        (out/'frame-names.json').write_text(json.dumps(dict(body_names=frame.body_names,joint_names=frame.joint_names),indent=2))
        colliders={}
        for side,name in [('left','LeftArm'),('right','RightArm')]:
            colliders[side]=collect(stage,'/World/envs/env_0/'+name,list(base.scene[side+'_robot'].body_names))
        (out/'gripper-colliders.json').write_text(json.dumps(colliders,indent=2))
        robot_geometry={}
        if req.get('frame_press',False) or req.get('robot_unlock',False):
            for side,name in [('left','LeftArm'),('right','RightArm')]:
                body_names=list(base.scene[side+'_robot'].body_names)
                robot_geometry[side]=collect(stage,'/World/envs/env_0/'+name,body_names,selected=body_names)
            (out/'robot-colliders.json').write_text(json.dumps(robot_geometry,indent=2))
            from robot_contacts import collect_cooked
            import hashlib
            cooked={}
            for side,name in [('left','LeftArm'),('right','RightArm')]:
                cooked[side]=collect_cooked(stage,'/World/envs/env_0/'+name,robot_geometry[side])
            cooked['_provenance']=dict(source_run=out.name,scene_modified=False,
                native_api='PhysX request_convex_collision_representation',
                source_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                    for name in ('scene.usda','robot-colliders.json','versions.json','manifest.json')})
            (out/'robot-cooked-colliders.json').write_text(json.dumps(cooked,indent=2))
        (out/'scene-audit.json').write_text(json.dumps(dict(laboratory_fixture=False,cloth=cover is not None,
            diagnostic_initialization=bool(req.get('release_probe_run')),
            cloth_attachments=[],cover_center_m=None if cover is None else cover.center.tolist(),
            frame_joints=constraints,actions=14,translation_S_W_m=shift.tolist(),
            table_center_m=(np.array(station.TABLE_CENTER)+shift).tolist(),table_size_m=station.TABLE_SIZE,robot='stock BimanualYamEmbodiment'),indent=2))
        from full_observation import make_observer
        bodies,observe=make_observer(base,frame,cover,colliders,robot_geometry,req)
        release_start = None
        if req.get('release_probe_run'):
            from release_probe import initialize
            release_start = initialize(base, frame, cover, req, out)
        session=RobotSession(env,out,req,observe)
        session.cover_center=None if cover is None else cover.center.copy()
        if release_start is not None:
            from release_probe import execute
            def read_release_frame():
                poses = bodies()
                return {name: transform(poses[frame.body_names.index(name)]) for name in ('Bridge', 'Handle')}
            from robot_reopen import transform
            execute(session, *release_start, read_release_frame)
            report.update(status='completed',termination_reason='release_probe_complete',
                control_steps=len(session.rows),physics_steps=session.physics_steps,
                limitations=['Cold initialization from a held pose, not a from-reset demonstration or independent episode replay. Benchmark scoring is forbidden.'])
            return
        if req.get('replay_run'):
            from episode_checks import replay
            replay(session,out.parent/req['replay_run'],out)
            report.update(status='completed',termination_reason='action_replay_complete',
                control_steps=len(session.rows),physics_steps=session.physics_steps,
                limitations=['Fresh simulation of ordinary recorded actions; independent contact and outcome audits still required.'])
            return
        if req.get('policy','scripted') in ('dream','interface-check'):
            from dream_evaluation import run_policy
            report.update(run_policy(session))
            return
        scripted_episode(session,frame,bodies,shift,table_top,metadata,robot_geometry,cover,report)
    except Exception:
        report.update(status='error',termination_reason='runtime_error',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    finally:
        if session:
            session.close()
            report.update(control_steps=len(session.rows),physics_steps=session.physics_steps)
        (out/'result.json').write_text(json.dumps(report,indent=2))
        if env:env.close()
        if app:app.close()


def scripted_episode(session,frame,bodies,shift,table_top,metadata,robot_geometry,cover,report):
    """Execute the same privileged scripted controller through ordinary session actions."""
    req,out=session.request,session.out
    if req.get('frame_press',False):
        from frame_press import press_sequence
        press_sequence(session,frame,bodies,shift,table_top,out,robot_geometry['right'])
        report.update(status='completed',termination_reason='diagnostic_complete',control_steps=len(session.rows),
            physics_steps=session.physics_steps,limitations=[
                'Free bare frame from a folded start; no cloth or full-task score.',
                'Robot press, lock, lift and release require independent contact and retention audit.'])
        return
    session.hold(1.,'settle_frame')
    pose=bodies()[frame.body_names.index('Handle')]
    axis=Rotation.from_quat(pose[3:]).as_matrix()[:,2]
    target=pose[:3]+.06*axis
    approach=np.array([.5,np.sqrt(.75),0.])
    closing=np.cross(approach,axis);closing/=np.linalg.norm(closing)
    pre=target-.07*approach
    lift_height=req.get('frame_lift')
    if lift_height is None:
        lift_height=.22 if req.get('upright_handle',False) else .14
    candidates=[]
    for sign in (1.,-1.):
        candidate=np.column_stack((sign*closing,np.cross(approach,sign*closing),approach))
        q0=session.action[7:13].copy();cost=0.
        try:
            for point in [pre+[0,0,.04],pre,target,target+[0,0,lift_height]]:
                q1,_,_=solve_pose(point-shift,candidate,q0,'right')
                cost+=float(np.sum((q1-q0)**2));q0=q1
            candidates.append((cost,sign,candidate))
        except AssertionError:
            pass
    assert candidates,'No reachable handle grasp orientation'
    _,sign,rotation=min(candidates,key=lambda item:item[0])
    pickup_rotation=rotation.copy()
    (out/'handle-grasp-selection.json').write_text(json.dumps(dict(closing_sign=sign,
        target_m=target.tolist(),handle_axis=axis.tolist(),costs=[(c[0],c[1]) for c in candidates]),indent=2))
    session.move('right',pre+np.array([0,0,.04]),rotation,.6,3.,'approach_handle',joint=True)
    session.move('right',pre,rotation,.6,1.,'align_handle')
    session.move('right',target,rotation,.6,2.,'enter_handle')
    session.move('right',target,rotation,.08,1.,'grasp_handle')
    session.hold(1.,'verify_handle_grasp')
    if req.get('robot_unlock',False):
        from robot_unlock import unlock_sequence
        unlock_sequence(session,frame,bodies,shift,out,target,rotation,lift_height)
        report.update(status='completed',termination_reason='controller_complete_pending_audit' if cover is not None else 'diagnostic_complete',control_steps=len(session.rows),
            physics_steps=session.physics_steps,limitations=[
                'Two-pocket robot integration; full outcome requires independent contact and scoring audit.' if cover is not None else
                'Bare-frame robot pedal release, inner-rail reopening and retention; no cloth or full-task score.' if req.get('robot_reopen',False) else 'Bare-frame robot pedal release and handle lift; no cloth or full-task score.',
                'Physical unlocking and all robot/frame contacts require independent audit.'])
        return
    initial_bridge=bodies()[frame.body_names.index('Bridge'),:3].copy()
    lifted_target=target+np.array([0,0,lift_height])
    session.move('right',lifted_target,rotation,.08,3.,'lift_flat_frame')
    if req.get('upright_handle',False):
        handle_pose=bodies()[frame.body_names.index('Handle')]
        current_axis=Rotation.from_quat(handle_pose[3:]).as_matrix()[:,2]
        cross=np.cross(current_axis,[0.,0.,1.]);sine=np.linalg.norm(cross)
        align=Rotation.from_rotvec(cross/max(sine,1e-12)*np.arctan2(sine,current_axis[2])).as_matrix()
        candidates=[]
        for yaw in (0.,45.,-45.,90.,-90.,180.):
            goal=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@align@rotation
            try:
                qgoal,_,_=solve_pose(lifted_target-shift,goal,session.action[7:13],'right')
                candidates.append((float(np.sum((qgoal-session.action[7:13])**2)),goal))
            except AssertionError:pass
        assert candidates,'Upright handle orientation unreachable'
        rotation=min(candidates,key=lambda item:item[0])[1]
        session.move('right',lifted_target,rotation,.08,3.,'turn_handle_upright')
        session.hold(1.,'settle_upright_handle')
    session.hold(2.,'hold_flat_frame')
    lifted_bridge=bodies()[frame.body_names.index('Bridge'),:3].copy()
    report['initial_handle_lift_m']=float(lifted_bridge[2]-initial_bridge[2])
    session.move('right',lifted_target+[0,.10,0],rotation,.08,2.,'translate_frame')
    session.move('right',lifted_target,rotation,.08,2.,'return_frame')
    session.hold(2.,'retention')
    place=target.copy()
    if req.get('upright_handle',False):
        # The upright wrist pose may be unreachable near the table even
        # though the lifted pose is reachable. Return to the already
        # validated pickup orientation before lowering the assembly.
        session.move('right',lifted_target,pickup_rotation,.08,3.,'return_handle_angle')
        session.hold(1.,'settle_returned_angle')
        rotation=pickup_rotation
        poses=bodies();minimum=float('inf')
        for shape in metadata['colliders']:
            if shape['body'] in ('Left','Right') and 'points' in shape:
                pose=poses[frame.body_names.index(shape['body'])]
                points=np.asarray(shape['points'])@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
                minimum=min(minimum,float(points[:,2].min()))
        place=lifted_target+np.array([0.,0.,table_top+.002-minimum])
    session.move('right',place,rotation,.08,3.,'set_down_frame')
    session.move('right',place,rotation,.6,1.,'release_handle')
    session.move('right',place-.08*rotation[:,2]+[0,0,.08],rotation,.6,2.,'withdraw')
    session.hold(3.,'released')
    report.update(status='completed',termination_reason='diagnostic_complete',control_steps=len(session.rows),
        physics_steps=session.physics_steps,limitations=['Free folding frame only; no textile or full-task score.',
        'Starts with a flat contact-locked frame; this does not test robot unlocking or pocket installation.',
        'Handle grasp and release need independent contact/retention audit.'])


if __name__=='__main__':main()
