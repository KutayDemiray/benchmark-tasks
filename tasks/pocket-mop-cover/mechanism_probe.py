"""Isolated force-limited press test of the estimated passive frame mechanism."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    request=parser.parse_args().request
    req=json.loads(request.read_text())
    out=request.parent
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    report=dict(status="initializing",diagnostic_only=True,full_task_success=False,
                mechanism_gate_passed=False)
    app=None
    camera=None
    writer=None
    video_frames=0
    visual_pose_error=0.
    visual_cloth_error=0.
    render_state_change=0.
    try:
        from isaaclab.app import AppLauncher
        kit_args = kit_arguments(out)
        app=AppLauncher(headless=True,enable_cameras=True,device="cuda:0",multi_gpu=False,kit_args=kit_args).app
        import torch
        import omni.usd
        from pxr import PhysxSchema,UsdPhysics,UsdLux
        from isaaclab.sim import SimulationCfg,SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.assets import Articulation,ArticulationCfg,DeformableObject,DeformableObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from native import box
        from cloth_solver import install as install_cloth_solver
        install_cloth_solver(req,out)
        from frame_solver import install
        install(req,out)
        from frame_native import author_frame,author_press_guide,passive_actuators,author_pedal_pusher
        from runtime_checks import verify_gpu
        sim=SimulationContext(SimulationCfg(device="cuda:0",dt=1/req["physics_hz"],
            render_interval=req["physics_hz"]//20,use_fabric=False,physics=PhysxCfg(),log_dir=str(out)))
        stage=omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api=PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr("GPU")
                api.CreateSolverTypeAttr("TGS")
        with_cover=req.get("with_cover",False)
        if with_cover and req.get('lab_camera',False):
            from isaaclab.sensors import Camera,CameraCfg
            from isaaclab.sim import PinholeCameraCfg
            camera=Camera(CameraCfg(prim_path='/World/ObservationCamera',update_period=0.,
                height=720,width=1280,data_types=['rgb'],
                spawn=PinholeCameraCfg(focal_length=18.,horizontal_aperture=20.955,
                    clipping_range=(.01,10.))))
        box(stage,"/World/Bench",(.8,.6,.04),(.22 if with_cover else 0.,0.,-.02))
        # Visible uprights identify the laboratory guide, away from the head.
        for i,y in enumerate((-.22,.22)):
            box(stage,"/World/Guide"+str(i),(.025,.025,.5),(.3,y,.25),color=(.7,.2,.15))
        with_pocket=req.get("with_pocket",False) or with_cover
        frame_origin=(.22 if with_cover else .218,0.,.170) if with_pocket else (0.,0.,.160)
        initial_angle=req.get('pocket_start_angle',55.)
        if with_pocket and initial_angle!=55.:
            angle=np.deg2rad(initial_angle)
            frame_origin=(frame_origin[0],0.,.192*np.sin(angle)+.003*np.cos(angle)+.004)
        metadata=author_frame(stage,origin=frame_origin,latch_contacts=not req.get("disable_latch_contact",False),snap_throat=req.get("snap_throat",False))
        pocket=None
        cloth_positions=[]
        cloth_velocities=[]
        if with_pocket:
            from mesh import pocket_mesh,material_rest,surface_edges,edge_strain
            from native import cloth
            from robot_gate import fabric
            from measurements import closed_pocket_faces,pocket_seating
            from scipy.spatial.transform import Rotation
            config=json.loads(Path(__file__).with_name("parameters.json").read_text())
            material=fabric(config,seam_height=.002 if req.get("finite_seam",False) else 0.)
            if req.get("bend_damping_rate") is not None:
                material["bend_damping"]=req["bend_damping_rate"]
            material.update(collision_pair_updates=req["contact_updates"],
                collision_subiterations=req["contact_subiterations"],speculative_ccd=req["ccd"])
            material.update(mesh_spacing_m=req["spacing"],initial_mouth_height_m=req["mouth_height"])
            if with_cover:
                from cover_mesh import cover_mesh
                initial,faces,cover=cover_mesh(material,spacing=req["spacing"],mouth_height=req["mouth_height"])
                pockets=cover["pockets"]
                markers=pockets[0]["markers"]
                (out/"cover-mesh.json").write_text(json.dumps(cover,indent=2))
            else:
                initial,faces,markers=pocket_mesh(material,spacing=req["spacing"],mouth_height=req["mouth_height"])
                pockets=[dict(markers=markers,faces=faces.tolist(),closed_faces=closed_pocket_faces(initial,faces,markers).tolist())]
            rest=material_rest(initial,material["flat_rest_panels"],material["seam_height_m"])
            edges=surface_edges(faces)
            closed=np.asarray(pockets[0]["closed_faces"])
            cloth(stage,initial,faces,material,req["iterations"])
            pocket=DeformableObject(DeformableObjectCfg(prim_path="/World/Pocket",spawn=None))
            (out/"effective-parameters.json").write_text(json.dumps(material,indent=2))
            (out/"mesh.json").write_text(json.dumps(dict(markers=markers),indent=2))
            np.savez_compressed(out/"geometry.npz",rest=rest,initial=initial,faces=faces)
            assert not any("Attachment" in p.GetTypeName() for p in stage.Traverse())
            (out/"scene-audit.json").write_text(json.dumps(dict(cloth_attachments=[],single_pocket_only=not with_cover,two_pockets=with_cover,
                laboratory_press=True,pocket_centre_m=[.10,0.,.002],frame_origin_m=frame_origin),indent=2))
        author_press_guide(stage,origin=metadata["origin"])
        travel=metadata["origin"][2]-.003+req.get("press_overtravel",0.)
        UsdLux.DomeLight.Define(stage,"/World/Light").CreateIntensityAttr(1800)
        rig=Articulation(ArticulationCfg(prim_path="/World/Frame",spawn=None,
            actuators={"laboratory_press":ImplicitActuatorCfg(joint_names_expr=["LaboratoryPress"],
                effort_limit_sim=15.,stiffness=500.,damping=8.),**passive_actuators()}))
        pusher=None
        if req.get("mechanism_unlock",False):
            assert req["seconds"]>=24+req.get("retention_extension",0.)
            metadata["pedal_pusher"]=author_pedal_pusher(stage,offset=(frame_origin[0],0.,0.))
            pusher=Articulation(ArticulationCfg(prim_path="/World/PedalPusher",spawn=None,
                actuators={"lab_pusher":ImplicitActuatorCfg(joint_names_expr=["Press"],
                    effort_limit_sim=20.,stiffness=800.,damping=10.)}))
        stage.GetRootLayer().Export(str(out/"scene.usda"))
        (out/"mechanism.json").write_text(json.dumps(metadata,indent=2))
        sim.reset()
        if camera is not None:
            camera.set_world_poses_from_view(torch.tensor([[.60,-.80,.70]],device='cuda:0'),
                torch.tensor([[.22,0.,.22]],device='cuda:0'))
            stage.GetRootLayer().Export(str(out/'scene.usda'))
        verify_gpu(out,sim,rig)
        def array(value):
            value=getattr(value,"torch",value)
            return value.detach().cpu().numpy().copy() if hasattr(value,"detach") else value.numpy().copy()
        (out/"native-joint-stiffness.json").write_text(json.dumps(dict(
            stiffness=array(rig.root_view.get_dof_stiffnesses()).tolist(),
            damping=array(rig.root_view.get_dof_dampings()).tolist()),indent=2))
        names=list(rig.joint_names)
        ids={name:names.index(name) for name in names}
        q=torch.zeros_like(rig.data.default_joint_pos.torch)
        q[0,ids["HingeLeft"]]=-np.deg2rad(initial_angle)
        q[0,ids["HingeRight"]]=np.deg2rad(initial_angle)
        q[0,ids["RockerHinge"]]=-np.deg2rad(20.)
        rig.write_joint_state_to_sim(q,torch.zeros_like(q))
        rig.reset()
        if pocket is not None:
            state=torch.zeros((1,len(initial),6),device="cuda:0")
            state[0,:,:3]=torch.as_tensor(initial,device="cuda:0")
            pocket.write_nodal_state_to_sim_index(state)
            pocket.reset()
            native_rest=array(pocket.root_view.get_rest_nodal_positions())[0].reshape(-1,3)[:len(rest)]
            assert np.allclose(native_rest,rest,atol=1e-7)
        report.update(status="running",joint_names=names)
        (out/"result.json").write_text(json.dumps(report,indent=2))
        rows=[]
        positions=[]
        body_states=[]
        pusher_states=[]
        dt=1/req["physics_hz"]
        with (out/"steps.jsonl").open("w",buffering=1) as log:
            for step in range(round(req["seconds"]/dt)):
                t=step*dt
                target=0.
                phase="folded"
                if 1<=t<4:
                    s=(t-1)/3
                    target=-travel*s*s*(3-2*s)
                    phase="press"
                elif 4<=t<6:
                    target=-travel
                    phase="flat_hold"
                elif 6<=t<9:
                    s=(t-6)/3
                    target=-travel+(travel+req.get('retention_raise',0.))*s*s*(3-2*s)
                    phase="lift"
                elif t>=9:
                    target=req.get('retention_raise',0.)
                    phase="retention"
                pusher_target=0.
                u=t-req.get("retention_extension",0.)
                if pusher is not None and u>=12:
                    floor_travel=metadata["origin"][2]-.003
                    if u<15:
                        s=(u-12)/3
                        target=-floor_travel*s*s*(3-2*s)
                        phase="set_down_for_unlock"
                    elif u<20:
                        target=-floor_travel+req.get("unlock_lift",0.)
                        phase="pedal_press"
                        if u<17:
                            s=(u-15)/2
                            pusher_target=-(.180+req.get("pedal_overtravel",0.))*s*s*(3-2*s)
                            target=-floor_travel+req.get("unlock_lift",0.)*s*s*(3-2*s)
                        elif u<18:
                            pusher_target=-(.180+req.get("pedal_overtravel",0.))
                        elif u<19:
                            s=u-18
                            pusher_target=-(.180+req.get("pedal_overtravel",0.))*(1-s*s*(3-2*s))
                        else:
                            phase="pedal_released"
                    elif u<23:
                        s=(u-20)/3
                        target=(-floor_travel+req.get("unlock_lift",0.))*(1-s*s*(3-2*s))
                        phase="lift_after_unlock"
                    else:
                        phase="unlocked_hold"
                    pusher.set_joint_position_target(torch.tensor([[pusher_target]],device="cuda:0"))
                    pusher.write_data_to_sim()
                rig.set_joint_position_target(torch.tensor([[target]],device="cuda:0",dtype=torch.float32),
                                              joint_ids=[ids["LaboratoryPress"]])
                rig.set_joint_position_target(torch.zeros((1,1),device="cuda:0"),joint_ids=[ids["ClipFlexure"]])
                rig.write_data_to_sim()
                sim.step(render=step%(req["physics_hz"]//20)==0)
                if camera is not None:
                    camera.update(dt)
                    if step%(req['physics_hz']//20)==0:
                        # Public PhysX output synchronization: native physics -> USD,
                        # never commanded poses or writes into the simulation state.
                        from omni.physx import get_physx_interface
                        from pxr import Usd,UsdGeom,PhysicsSchemaTools
                        before=array(rig.root_view.get_link_transforms())[0].reshape(-1,7)
                        cloth_before=array(pocket.root_view.get_simulation_nodal_positions())[0].reshape(-1,3)[:len(initial)]
                        get_physx_interface().update_transformations(False,True,True)
                        for prim in stage.Traverse():
                            if prim.IsA(UsdPhysics.Scene):
                                get_physx_interface().update_transformations_scene(
                                    PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),True,True)
                        after=array(rig.root_view.get_link_transforms())[0].reshape(-1,7)
                        cloth_after=array(pocket.root_view.get_simulation_nodal_positions())[0].reshape(-1,3)[:len(initial)]
                        render_state_change=max(render_state_change,float(abs(after-before).max()),float(abs(cloth_after-cloth_before).max()))
                        assert render_state_change<1e-10, 'Rendering synchronization changed native state'
                        cache=UsdGeom.XformCache()
                        for index,name in enumerate(rig.body_names):
                            matrix=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Frame/'+name)))
                            visual_pose_error=max(visual_pose_error,float(np.linalg.norm(matrix[3,:3]-after[index,:3])))
                            rotation=Rotation.from_quat(after[index,3:]).as_matrix()
                            rotation_error=float(np.max(abs(matrix[:3,:3].T-rotation)))
                            if rotation_error>=2e-5:
                                body_prim=stage.GetPrimAtPath('/World/Frame/'+name)
                                (out/'visual-sync-error.json').write_text(json.dumps(dict(
                                    name=name,body_names=rig.body_names,backend_body_names=rig.backend_body_names,
                                    rotation_error=rotation_error,native=after[index].tolist(),
                                    usd_default=matrix.tolist(),
                                    usd_time_zero=np.asarray(UsdGeom.XformCache(Usd.TimeCode(0.)).GetLocalToWorldTransform(body_prim)).tolist(),
                                    data_pose=array(rig.data.body_link_pose_w)[0,index].tolist(),
                                    xform_ops={str(op.GetAttr().GetName()):dict(default=str(op.Get()),times=op.GetAttr().GetTimeSamples())
                                        for op in UsdGeom.Xformable(body_prim).GetOrderedXformOps()}),indent=2))
                            assert rotation_error<2e-5, 'Stale rigid visual orientation'
                        points=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath('/World/Pocket')).GetPointsAttr().Get())
                        visual_cloth_error=max(visual_cloth_error,float(np.linalg.norm(points-cloth_after,axis=1).max()))
                        assert max(visual_pose_error,visual_cloth_error)<2e-5, 'Stale rendered geometry'
                        rgb=array(camera.data.output['rgb'])[0,:,:,:3].astype(np.uint8)
                        if writer is None:
                            writer=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo',
                                '-pix_fmt','rgb24','-s','1280x720','-r','20','-i','-',
                                '-c:v','libx264','-threads','2','-preset','fast','-crf','22',
                                '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'lab_cam.mp4')],
                                stdin=subprocess.PIPE)
                        writer.stdin.write(rgb.tobytes())
                        video_frames+=1
                        if step%req['physics_hz']==0:
                            from PIL import Image
                            Image.fromarray(rgb).save(out/'lab_cam-latest.png')
                rig.update(dt)
                if pusher is not None:
                    pusher.update(dt)
                    pusher_states.append(array(pusher.data.body_link_pose_w)[0])
                joints=array(rig.data.joint_pos)[0]
                bodies=array(rig.data.body_link_pose_w)[0]
                assert np.isfinite(joints).all() and np.isfinite(bodies).all()
                wing=np.rad2deg(joints[[ids["HingeLeft"],ids["HingeRight"]]])
                row=dict(step=step,time_s=t+dt,phase=phase,press_target_m=target,
                    joint_positions=joints.tolist(),wing_angles_deg=wing.tolist(),
                    actuator_position_targets=array(rig.actuators.target_command.position)[0].tolist(),
                    actuator_computed_effort=array(rig.actuators.computed_effort)[0].tolist(),
                    clip_deflection_deg=float(np.rad2deg(joints[ids["ClipFlexure"]])),
                    rocker_angle_deg=float(np.rad2deg(joints[ids["RockerHinge"]])),
                    gear_error_rad=float(abs(joints[ids["HingeLeft"]]+joints[ids["HingeRight"]])),
                    score=0.,robot_action=None,pedal_pusher_target_m=pusher_target,
                    pedal_pusher_joint_m=float(array(pusher.data.joint_pos)[0,0]) if pusher is not None else None)
                if pocket is not None:
                    pocket.update(dt)
                    x=array(pocket.root_view.get_simulation_nodal_positions())[0].reshape(-1,3)[:len(rest)]
                    v=array(pocket.root_view.get_simulation_nodal_velocities())[0].reshape(-1,3)[:len(rest)]
                    cloth_positions.append(x)
                    cloth_velocities.append(v)
                    row.update(max_abs_edge_strain=float(abs(edge_strain(rest,x,faces,edges=edges)).max()),
                        max_cloth_speed_m_s=float(np.linalg.norm(v,axis=1).max()),min_cloth_z_m=float(x[:,2].min()),
                        mouth_median_height_m=float(np.median(x[markers["mouth_upper"],2]-x[markers["mouth_lower"],2])))
                    assert np.isfinite(x).all() and np.isfinite(v).all()
                    if step%(req["physics_hz"]//20)==0:
                        left_pose=bodies[rig.body_names.index("Left")]
                        rotation=Rotation.from_quat(left_pose[3:]).as_matrix()
                        end_centre=left_pose[:3]+rotation@np.array([-.142,0.,0.])
                        row["pocket_seating"]=pocket_seating(x,closed,end_centre,[.1,.1,.006],
                            markers=markers,faces=np.asarray(pockets[0]["faces"]),profile="beveled",rotation=rotation)
                        if with_cover:
                            right_pose=bodies[rig.body_names.index("Right")]
                            right_rotation=Rotation.from_quat(right_pose[3:]).as_matrix()
                            right_centre=right_pose[:3]+right_rotation@np.array([.142,0.,0.])
                            other=pockets[1]
                            row["right_pocket_seating"]=pocket_seating(x,np.asarray(other["closed_faces"]),right_centre,[.1,.1,.006],
                                markers=other["markers"],faces=np.asarray(other["faces"]),profile="beveled",
                                rotation=right_rotation@np.diag([-1.,-1.,1.]))
                            row["both_pockets_seated"]=row["pocket_seating"]["seated_samples"] and row["right_pocket_seating"]["seated_samples"]
                log.write(json.dumps(row)+"\n")
                rows.append(row)
                positions.append(joints)
                body_states.append(bodies)
                if step%req["physics_hz"]==0:
                    (out/"heartbeat.json").write_text(json.dumps(row,indent=2))
                    print(json.dumps(row),flush=True)
        np.savez_compressed(out/"states.npz",joint_positions=positions,body_poses=body_states,
                            joint_names=names,body_names=rig.body_names,
                            pusher_poses=pusher_states,pusher_body_names=pusher.body_names if pusher is not None else [])
        if pocket is not None:
            np.savez_compressed(out/"cloth-states.npz",positions=cloth_positions,velocities=cloth_velocities)
        hold=[r for r in rows if r["phase"]=="retention" and 10<=r["time_s"]<=12+req.get("retention_extension",0.)]
        flat=max(abs(v) for r in hold for v in r["wing_angles_deg"]) if hold else 999.
        report.update(status="completed",termination_reason="diagnostic_time_limit",steps=len(rows),
            retention_window_seconds=[10,12+req.get("retention_extension",0.)],
            retained_window_max_wing_angle_deg=flat,last_two_seconds_max_wing_angle_deg=flat,
            last_two_seconds_max_gear_error_rad=max(r["gear_error_rad"] for r in hold) if hold else None,
            passive_retention_observed=bool(hold and flat<3.),
            physical_unlock_observed=bool(pusher is not None and min(abs(v) for r in rows[-req["physics_hz"]:] for v in r["wing_angles_deg"])>30.),
            mechanism_gate_passed=False,
            limitations=["Two-pocket laboratory installation; no robot or benchmark score." if with_cover else "Single-pocket folding-frame laboratory test; no robot or second pocket." if with_pocket else "Rigid-frame laboratory test only; no cloth or robot.",
                "Positive retention must be paired with a disabled-contact negative control and contact audit.",
                "Mechanism dimensions are estimates from a related manufacturer mechanism family."])
    except Exception:
        report.update(status="error",termination_reason="runtime_error",traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        if writer is not None:
            writer.stdin.close()
            encoded=writer.wait()
            decoded=subprocess.run(['ffmpeg','-v','error','-i',str(out/'lab_cam.mp4'),
                '-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True)
            counts=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
            count=counts[-1] if counts else 0
            passed=encoded==0 and decoded.returncode==0 and count==video_frames==round(req['seconds']*20)
            (out/'video-verification.json').write_text(json.dumps(dict(passed=passed,
                decoded_frames=count,expected_frames=round(req['seconds']*20),recorded_frames=video_frames,
                scope='Actual native laboratory camera; force-limited fixture, no robot or full-task score.',
                decoder_error=decoded.stderr,visual_pose_max_error_m=visual_pose_error,
                visual_cloth_max_error_m=visual_cloth_error,native_state_change_from_render_sync=render_state_change),indent=2))
        (out/"result.json").write_text(json.dumps(report,indent=2))
        if app:
            app.close()


if __name__=="__main__":
    main()
