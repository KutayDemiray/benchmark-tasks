"""Stock-YAM cloth grasp diagnostic; no full task or cloth attachments."""

from task_paths import OUTPUT_ROOT

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from mesh import audit, edge_strain, pocket_mesh, surface_edges, material_rest


def array(value):
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return value.numpy().copy() if hasattr(value, "numpy") else np.asarray(value).copy()


def fabric(config, seam_height=0.):
    p = dict(config["estimated_probe"])
    alternative = config["alternative_product"]
    e = alternative["estimated"]
    area = e["cover_width_m"] * (e["cover_length_m"] + 2 * e["pocket_depth_m"])
    area+=2*(e["cover_width_m"]+2*e["pocket_depth_m"])*seam_height
    p["seam_height_m"]=seam_height
    p["flat_rest_panels"]=bool(seam_height)
    p["density_kg_m3"] = alternative["published"]["cover_dry_mass_kg"] / (area * p["shell_thickness_m"])
    return p


def surface_asset(rest, faces, material, iterations):
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.asset import Asset

    def spawn(path, cfg, translation=None, orientation=None, **kwargs):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import PhysxSchema, UsdPhysics
        from native import cloth
        stage = get_current_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr("GPU")
                api.CreateSolverTypeAttr("TGS")
        return cloth(stage, rest, faces, material, iterations, path=path).GetPrim()

    def reset(env, env_ids):
        import torch
        obj = env.scene["pocket"]
        ids = env_ids
        origin = getattr(env.scene.env_origins, "torch", env.scene.env_origins)
        initial = torch.zeros((len(ids), len(rest), 6), device=env.device)
        initial[:, :, :3] = torch.as_tensor(rest, device=env.device)[None] + origin[ids, None]
        obj.write_nodal_state_to_sim_index(initial, env_ids=ids)
        obj.reset(env_ids=ids)

    @configclass
    class SpawnCfg(SpawnerCfg):
        func: object = clone(spawn)

    class Surface(Asset):
        def __init__(self):
            super().__init__("pocket", tags=["surface-deformable", "cloth"])
        def get_object_cfg(self):
            return self.name, DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/Pocket", spawn=SpawnCfg())
        def get_event_cfg(self):
            return self.name, EventTermCfg(func=reset, mode="reset")
    return Surface()


def solve_pose(position, rotation, q0, side):
    from scipy.optimize import least_squares
    from yam_arena import ik, station
    def error(q):
        p, r = ik.tcp_pose(q, station.BASES[side])
        return np.r_[p - position, 0.3 * ik._rot_error(rotation, r)]
    best = None
    seeds = [q0, *np.random.default_rng(0).uniform(ik.LOWER, ik.UPPER, (8, 6))]
    for seed in seeds:
        result = least_squares(error, seed, bounds=(ik.LOWER, ik.UPPER), max_nfev=160)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
        if np.linalg.norm(best.fun[:3]) < 0.001 and np.linalg.norm(best.fun[3:]) < 0.003:
            break
    pe = float(np.linalg.norm(best.fun[:3]))
    re = float(np.linalg.norm(best.fun[3:]) / 0.3)
    assert pe < 0.004 and re < 0.04, f"Unreachable {side} target {position}: {pe} m, {re} rad"
    return best.x, pe, re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    req_path = parser.parse_args().request
    req = json.loads(req_path.read_text())
    out = req_path.parent
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    assert out.resolve().is_relative_to(OUTPUT_ROOT)
    assert req["product_profile"] == "base_micro" and not req["reinforced"]
    config = json.loads(Path(__file__).with_name("parameters.json").read_text())
    material = fabric(config,seam_height=.002 if req.get("finite_seam",False) else 0.)
    material["flat_rest_panels"]=bool(req.get("flat_rest_panels",False) or req.get("finite_seam",False))
    material["flat_rest_bending"]=req.get("flat_rest_bending",False)
    if req.get("bend_damping_rate") is not None:
        material["bend_damping"]=req["bend_damping_rate"]
    material.update(collision_pair_updates=req.get("contact_updates",1),
        collision_subiterations=req.get("contact_subiterations",1),speculative_ccd=req.get("ccd",False))
    material.update(mesh_spacing_m=req["spacing"],initial_mouth_height_m=req["mouth_height"])
    (out/"effective-parameters.json").write_text(json.dumps(material,indent=2))
    local, faces, markers = pocket_mesh(material, spacing=req["spacing"], mouth_height=req["mouth_height"])
    cloth_edges = surface_edges(faces)
    app = env = None
    report = dict(status="initializing", diagnostic_only=True, full_task_success=False, product_gate_passed=False)
    writers = {}
    log = None
    try:
        from isaaclab.app import AppLauncher
        kit_args = kit_arguments(out)
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", multi_gpu=False, kit_args=kit_args).app
        import torch
        import omni.usd
        from cloth_solver import install as install_cloth_solver
        install_cloth_solver(req,out)
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from yam_arena import ik, station
        from yam_arena.constants import READY, SHOULDER, WRIST, GRIPPER, FINGER_OPEN
        from yam_arena.actions import bimanual_joint_pos_14
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import _box, env_cfg_callback, task_room
        from runtime_checks import verify_gpu

        from station_coordinates import TRANSLATION_S_W,translated_rig,translated_room,verify_station
        shift=TRANSLATION_S_W.copy() if req.get("table_origin",False) else np.zeros(3)
        table_top=station.TABLE_TOP+shift[2]
        support_top = table_top + 0.12
        rest = local + np.array([0.20+shift[0], 0.10+shift[1], support_top])
        metric_rest=material_rest(rest,material["flat_rest_panels"],material["seam_height_m"])
        # A visible rectangular laboratory support exposes the cloth's margins.
        # It does not guide insertion, pin nodes or retain the cloth.
        wide_block=req.get("placement")=="bench"
        support_size=(.40,.10,.12) if wide_block else (.25,.10,.12)
        support_center=np.array([.375 if wide_block else .30,.10,station.TABLE_TOP+.06])+shift
        support = _box("bench_block",support_size,support_center,(0.5,0.48,0.4))
        stock_rig=station.rig()
        embodiment = BimanualYamEmbodiment(enable_cameras=True, rig=translated_rig(stock_rig,shift))
        base_configure = env_cfg_callback(20)
        def configure(cfg):
            cfg = base_configure(cfg)
            cfg.sim.dt = 1 / req["physics_hz"]
            cfg.decimation = req["physics_hz"] // 20
            cfg.sim.render_interval = cfg.decimation
            cfg.sim.use_fabric = False
            cfg.sim.log_dir = str(out)
            return cfg
        definition = IsaacLabArenaEnvironment(
            name="pocket_mop_single_grasp", scene=Scene([*translated_room(stock_rig,shift),
                support, surface_asset(rest, faces, material, req["iterations"])]),
            embodiment=embodiment, env_cfg_callback=configure)
        env = ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device="cuda:0")).make_registered()
        base = env.unwrapped
        env.reset(seed=req["seed"])
        obj = base.scene["pocket"]
        verify_gpu(out, base.sim, obj)
        from capture_yam_contract import capture as capture_yam_contract
        capture_yam_contract(base, out)
        stage = omni.usd.get_context().get_stage()
        native_station=verify_station(stage,stock_rig,shift,{"bench_block":support_center})
        (out/"station-pose-check.json").write_text(json.dumps(native_station,indent=2))
        from robot_contacts import collect
        rigid_geometry={}
        for side,prim_name in [("left","LeftArm"),("right","RightArm")]:
            rigid_geometry[side]=collect(stage,"/World/envs/env_0/"+prim_name,
                                        list(base.scene[side+"_robot"].body_names))
        (out/"gripper-colliders.json").write_text(json.dumps(rigid_geometry,indent=2))
        attachments = [str(p.GetPath()) for p in stage.Traverse() if "Attachment" in p.GetTypeName()]
        assert not attachments, "Grasp must arise exclusively from contact"
        (out / "scene-audit.json").write_text(json.dumps({"cloth_attachments": attachments,
            "support": "Visible laboratory support block", "support_size_m":support_size, "support_center_m":support_center.tolist(), "table_center_m":(np.array(station.TABLE_CENTER)+shift).tolist(), "table_size_m":station.TABLE_SIZE, "translation_S_W_m":shift.tolist(), "robot": "stock BimanualYamEmbodiment", "actions": 14}, indent=2))
        stage.GetRootLayer().Export(str(out / "scene.usda"))
        (out / "mesh.json").write_text(json.dumps({**audit(rest, faces), "markers": markers}, indent=2))
        np.savez_compressed(out / "geometry.npz", rest=metric_rest, initial=rest, faces=faces)
        from robot_contacts import collect_cooked
        import hashlib
        from pxr import Usd,UsdUtils
        source_out=out
        source_stage=stage
        source_geometry=rigid_geometry
        if req.get("cooked_collider_probe",False) and req.get("state_run"):
            source_out=out.parent/req["state_run"]
            assert source_out.resolve().parent==out.parent.resolve()
            assert json.loads((source_out/"result.json").read_text())["status"]=="completed"
            assert json.loads((source_out/"versions.json").read_text())==json.loads((out/"versions.json").read_text())
            source_stage=Usd.Stage.Open(str(source_out/"scene.usda"))
            UsdUtils.StageCache.Get().Insert(source_stage)
            source_geometry=json.loads((source_out/"gripper-colliders.json").read_text())
        cooked={}
        for side,prim_name in [("left","LeftArm"),("right","RightArm")]:
            cooked[side]=collect_cooked(source_stage,"/World/envs/env_0/"+prim_name,source_geometry[side])
        cooked["_provenance"]=dict(source_run=source_out.name,cooking_probe_run=out.name,
            source_sha256={name:hashlib.sha256((source_out/name).read_bytes()).hexdigest()
                for name in ("scene.usda","gripper-colliders.json","versions.json","manifest.json")},
            native_api="PhysX request_convex_collision_representation",scene_modified=False)
        (out/"gripper-cooked-colliders.json").write_text(json.dumps(cooked,indent=2))
        if req.get("cooked_collider_probe",False):
            report.update(status="completed",termination_reason="read_only_cooked_geometry_export",
                control_steps=0,physics_steps=0,geometry_export_only=True,
                limitations=["Read-only stock collider cooking diagnostic, not a grasp or physics qualification."])
            return
        chunk_dir = out / "physics-chunks"
        chunk_dir.mkdir()
        chunk_pos, chunk_vel = [], []
        chunk_rigid={"left":[],"right":[]}
        physics_count = 0
        chunk_count = 0
        max_strain = 0.0
        def flush_chunk():
            nonlocal chunk_count
            if chunk_pos:
                target = chunk_dir / f"{chunk_count:05d}.npz"
                np.savez_compressed(target, positions=chunk_pos, velocities=chunk_vel,
                                    physics_end_step=physics_count,
                                    left_grasp_poses=chunk_rigid["left"],right_grasp_poses=chunk_rigid["right"])
                chunk_count += 1
                chunk_pos.clear()
                chunk_vel.clear()
                for values in chunk_rigid.values():values.clear()
        original_step = base.sim.step
        def recorded_step(*args, **kwargs):
            nonlocal physics_count, max_strain
            result = original_step(*args, **kwargs)
            positions = array(obj.root_view.get_simulation_nodal_positions())[0].reshape(-1, 3)[:len(rest)]
            velocities = array(obj.root_view.get_simulation_nodal_velocities())[0].reshape(-1, 3)[:len(rest)]
            assert np.isfinite(positions).all() and np.isfinite(velocities).all(), "Nonfinite cloth"
            chunk_pos.append(positions)
            chunk_vel.append(velocities)
            for side in ("left","right"):
                robot=base.scene[side+"_robot"]
                poses=array(robot.root_view.get_link_transforms())[0].reshape(-1,7)
                chunk_rigid[side].append(poses[rigid_geometry[side]["indices"]])
            physics_count += 1
            max_strain = max(max_strain, float(abs(edge_strain(metric_rest, positions, faces, edges=cloth_edges)).max()))
            if len(chunk_pos) >= req["physics_hz"]:
                flush_chunk()
            return result
        base.sim.step = recorded_step
        from PIL import Image
        import subprocess
        log = (out / "steps.jsonl").open("w", buffering=1)
        action = np.r_[READY, 1., READY, 1.]
        records = []
        def nodes():
            return array(obj.root_view.get_simulation_nodal_positions())[0].reshape(-1, 3)[:len(rest)]
        def step(target, phase):
            nonlocal action
            assert len(records) < round(req["seconds"]*20), "Diagnostic control limit"
            previous = action.copy()
            action = np.asarray(target, dtype=float).copy()
            assert action.shape == (14,)
            limits = np.tile([SHOULDER["velocity"]/20]*3+[WRIST["velocity"]/20]*3+
                             [GRIPPER["velocity"]/abs(FINGER_OPEN)/20], 2)
            assert np.all(abs(action-previous) <= limits*1.001), "Action exceeds stock velocity limits"
            for offset in (0,7):
                assert np.all(action[offset:offset+6] >= ik.LOWER)
                assert np.all(action[offset:offset+6] <= ik.UPPER)
                assert 0 <= action[offset+6] <= 1
            env.step(torch.tensor(action[None], device=base.device, dtype=torch.float32))
            points = nodes()
            observed = array(bimanual_joint_pos_14(base))[0]
            row = dict(step=len(records), time_s=(len(records)+1)/20, phase=phase,
                       robot_action=action.tolist(), measured_joint_pos=observed.tolist(),
                       max_abs_edge_strain=max_strain, min_cloth_z_m=float(points[:,2].min()),
                       cloth_center_m=points.mean(0).tolist(), score=0., full_task_success=False)
            records.append(row)
            log.write(json.dumps(row)+"\n")
            for camera in ("top_cam", "left_cam", "right_cam"):
                rgb = array(base.scene.sensors[camera].data.output["rgb"])[0,:,:,:3].astype(np.uint8)
                if camera not in writers:
                    h,w=rgb.shape[:2]
                    writers[camera] = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                        "-s",f"{w}x{h}","-r","20","-i","-","-c:v","libx264","-threads","2","-preset","ultrafast",
                        "-crf","23","-pix_fmt","yuv420p",str(out/(camera+".mp4"))],stdin=subprocess.PIPE)
                writers[camera].stdin.write(rgb.tobytes())
                if len(records)%20 == 0:
                    Image.fromarray(rgb).save(out/(camera+"-latest.png"))
            if len(records)%20 == 0:
                (out/"heartbeat.json").write_text(json.dumps(row,indent=2))
                print(json.dumps(row),flush=True)
            assert max_strain < 2., "Catastrophic cloth distortion"
        def hold(seconds, phase):
            for _ in range(round(seconds*20)):
                step(action, phase)
        plan = []
        def move(position, rotation, grip, seconds, phase, joint=False, right_position=None):
            goals=[("left",0,np.asarray(position),rotation)]
            if req.get("two_hands",False):
                mirrored=right_target+(np.asarray(position)-reference_left)*np.array([1.,-1.,1.])
                goals.append(("right",7,mirrored if right_position is None else np.asarray(right_position),right_rotation))
            paths=[]
            for side,offset,goal,goal_rotation in goals:
                q0=action[offset:offset+6].copy()
                p0,r0=ik.tcp_pose(q0,station.BASES[side])
                p0=p0+shift
                qgoal,pe,re=solve_pose(goal-shift,goal_rotation,q0,side)
                paths.append(dict(side=side,offset=offset,q0=q0,q=q0.copy(),p0=p0,r0=r0,
                                  goal=goal,rotation=goal_rotation,qgoal=qgoal,g0=action[offset+6]))
                plan.append(dict(phase=phase,side=side,position=goal.tolist(),rotation=goal_rotation.tolist(),
                                 ik_error_m=pe,ik_error_rad=re,grip=grip))
            (out/"controller-plan.json").write_text(json.dumps(plan,indent=2))
            count=round(seconds*20)
            for k in range(1,count+1):
                s=ik.smoothstep(k/count)
                command=action.copy()
                for path in paths:
                    if joint:
                        q=path["q0"]+s*(path["qgoal"]-path["q0"])
                    else:
                        p=path["p0"]+s*(path["goal"]-path["p0"])
                        r=ik.slerp_matrix(path["r0"],path["rotation"],s)
                        q,pe,re=ik.solve(p-shift,ik.matrix_to_quat_xyzw(r),path["q"],
                            base_pos=station.BASES[path["side"]],iters=120)
                        assert pe<.004 and re<.04, f"Unreachable {path['side']} path {phase}: {pe}, {re}"
                    path["q"]=q
                    offset=path["offset"]
                    command[offset:offset+6]=q
                    command[offset+6]=path["g0"]+s*(grip-path["g0"])
                step(command,phase)
        report["status"]="running"
        (out/"result.json").write_text(json.dumps(report,indent=2))
        hold(2.,"settle")
        points = nodes()
        patch = np.flatnonzero((local[:,0]>.135)&(local[:,0]<.155)&(local[:,1]>.055))
        center = points[patch].mean(0)
        # Approach the exposed side margin. A diagonal approach stays within
        # the stock wrist limits; the closing axis is normal to the textile.
        approach=np.array([.5,-np.sqrt(.75),0.])
        closing=np.array([0.,0.,1.])
        rotation=np.column_stack((closing,np.cross(approach,closing),approach))
        target=center+0.006*approach
        target[2]+=req.get("left_grasp_height",0.)
        target[1]+=req.get("grasp_margin_clearance",0.)
        reference_left=target.copy()
        right_patch=np.flatnonzero((local[:,0]>.135)&(local[:,0]<.155)&(local[:,1]<-.055))
        right_approach=approach*np.array([1.,-1.,1.])
        right_target=points[right_patch].mean(0)+.006*right_approach
        right_target[2]+=req.get("right_grasp_height",0.)
        right_target[1]-=req.get("grasp_margin_clearance",0.)
        right_rotation=np.column_stack((closing,np.cross(right_approach,closing),right_approach))
        pre=target-.06*approach
        (out/"grasp-observation.json").write_text(json.dumps({"patch_ids":patch.tolist(),"center":center.tolist(),
            "target":target.tolist(),"right_patch_ids":right_patch.tolist(),"right_target":right_target.tolist(),"two_hands":req.get("two_hands",False),"read_only_native_state":True},indent=2))
        move(pre+np.array([0.,0.,.10]),rotation,.35,3.,"approach_above",joint=True)
        move(pre,rotation,.35,2.,"approach_margin")
        move(target,rotation,.35,2.,"enter_margin")
        grasp_opening=req.get("grasp_opening",.008)
        close_seconds=req.get("close_seconds",1.)
        move(target,rotation,grasp_opening,close_seconds,"close")
        if close_seconds>1.:
            hold(1.,"settle_closed_grasp")
        grasp_origin=nodes()[patch].mean(0)
        lift_height=.13 if req.get("placement","bench")=="bench" else .22
        move(target+np.array([0.,0.,lift_height]),rotation,grasp_opening,3.,"lift")
        hold(1.,"settle_grasp")
        hold(2.,"hold_grasp")
        lifted=nodes()[patch].mean(0)
        report["patch_lift_m"]=float(lifted[2]-grasp_origin[2])
        report["cloth_min_height_over_block_m"]=float(nodes()[:,2].min()-support_top)
        if req.get("placement_tilt",0.):
            angle=np.deg2rad(req["placement_tilt"])
            def pitched(direction):
                forward=direction*np.cos(angle)-np.array([0.,0.,np.sin(angle)])
                normal=np.array([0.,0.,np.cos(angle)])+direction*np.sin(angle)
                return forward,np.column_stack((normal,np.cross(forward,normal),forward))
            approach,rotation=pitched(approach)
            right_approach,right_rotation=pitched(right_approach)
            move(target+np.array([0.,0.,lift_height]),rotation,grasp_opening,2.,"tilt_for_placement",
                right_position=right_target+np.array([0.,0.,lift_height]))
        place=target.copy()
        if req.get("placement","bench")=="table":
            move(target+np.array([.03,.15,lift_height]),rotation,grasp_opening,3.,"carry_clear_of_block",
                right_position=right_target+np.array([.03,.15,lift_height]))
            place=np.array([target[0]+.10,target[1]+.15,table_top+.025])
        if req["placement"]=="forward_table":
            assert req.get("two_hands",False), "Forward lay-down diagnostic requires the optional two-margin grasp"
            move(target+np.array([.13,0.,lift_height]),rotation,grasp_opening,3.,"carry_forward",
                right_position=right_target+np.array([.13,0.,lift_height]))
            early_feed=req.get("early_feed",False)
            edge=target+np.array([.13,0.,table_top+(.19 if early_feed else .15)-target[2]])
            move(edge,rotation,grasp_opening,3. if early_feed else 2.,"touch_free_edge",
                right_position=right_target+(edge-target))
            place=target+np.array([.27 if early_feed else .26,0.,table_top+.025-target[2]])
        if req["placement"]=="bench":
            move(target+np.array([0.,0.,.06]),rotation,grasp_opening,1.,"lower_cloth_edge")
            place=np.array([target[0]+.10,target[1],support_top+.025])
        right_place=right_target+(place-target)
        move(place,rotation,grasp_opening,4. if req.get("early_feed",False) else 3.,"lower_to_support",right_position=right_place)
        hold(1.,"place_cloth")
        release_opening=req.get("release_opening",.25)
        move(place,rotation,release_opening,1.,"release",right_position=right_place)
        if req.get("two_hands",False) or req["placement"]=="bench":
            move(place-.06*approach,rotation,release_opening,1.,"withdraw_fingers",
                right_position=right_place-.06*right_approach)
        retreat_delta=np.array([-.10,0.,.15]) if req.get("placement_tilt",0.) else np.array([0.,0.,.20])
        if req["placement"]=="table":
            retreat_delta=np.array([-.05,-.05,.15])
        retreat=place+retreat_delta-.06*approach
        right_retreat=right_place+retreat_delta-.06*right_approach
        move(retreat,rotation,release_opening,2.,"retreat",right_position=right_retreat)
        move(retreat,rotation,1.,1.,"open_clear_of_support",right_position=right_retreat)
        hold(5.,"released")
        flush_chunk()
        report.update(status="completed", termination_reason="diagnostic_complete", control_steps=len(records),
                      physics_steps=physics_count, max_abs_edge_strain=max_strain,
                      grasp_observed=report["patch_lift_m"]>.08,
                      numerical_gate_passed=max_strain<=config["gate"]["max_abs_edge_strain"],
                      limitations=["Single-pocket grasp diagnostic only; no frame or insertion.",
                          "Bench block is an explicit laboratory support; it does not prove the full-task starting layout."])
    except Exception:
        report.update(status="error",termination_reason="runtime_error",traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        if "flush_chunk" in locals():
            flush_chunk()
        if log:
            log.close()
        for writer in writers.values():
            writer.stdin.close()
            writer.wait()
        (out/"result.json").write_text(json.dumps(report,indent=2))
        if env:
            env.close()
        if app:
            app.close()


if __name__ == "__main__":
    main()
