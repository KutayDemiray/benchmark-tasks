"""Contact-only coupon experiment, explicitly not a YAM or full task demo."""

from task_paths import OUTPUT_ROOT

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

import numpy as np

from mesh import audit, edge_strain, extrude_surface, pocket_mesh, split_lip, surface_edges, material_rest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    req = json.loads(args.request.read_text())
    req.setdefault("lip_model", "surface")
    req.setdefault("blade_profile", "square")
    from blade_geometry import node_penetration, triangle_penetration
    req.setdefault("product_profile", "sprint")
    req.setdefault("seam_filter", "element")
    req.setdefault("detach_lip", False)
    req.setdefault("filter_grouping", "pairs")
    out = args.request.parent
    assert out.resolve().is_relative_to(OUTPUT_ROOT), "Output ownership"
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    config = json.loads(Path(__file__).with_name("parameters.json").read_text())
    p = dict(config["estimated_probe"])
    start_x = config["diagnostic_rig"]["blade_initial_center_x_m"]
    end_x = config["diagnostic_rig"]["blade_final_center_x_m"]
    travel_seconds = (start_x-end_x)/p["blade_speed_m_s"]
    hold_start, withdraw_start = 1+travel_seconds, 3+travel_seconds
    released_start = withdraw_start+travel_seconds
    if req["product_profile"] == "base_micro":
        alternative = config["alternative_product"]
        estimate = alternative["estimated"]
        projected_area = estimate["cover_width_m"] * (estimate["cover_length_m"] + 2 * estimate["pocket_depth_m"])
        p["density_kg_m3"] = alternative["published"]["cover_dry_mass_kg"] / (projected_area * p["shell_thickness_m"])
        assert not req["reinforced"], "Single-material polyester profile"
    p.update(collision_pair_updates=req.get("contact_updates",1),
             collision_subiterations=req.get("contact_subiterations",1))
    if req.get("finite_seam",False):
        from robot_gate import fabric
        p=fabric(config,seam_height=.002)
        p.update(collision_pair_updates=req.get("contact_updates",1),
            collision_subiterations=req.get("contact_subiterations",1),speculative_ccd=req.get("ccd",False))
    if req.get("bend_damping_rate") is not None:
        p["bend_damping"]=req["bend_damping_rate"]
    p.update(mesh_spacing_m=req["spacing"],initial_mouth_height_m=req["mouth_height"])
    config["effective_probe"] = p
    (out / "effective-parameters.json").write_text(json.dumps(config, indent=2))
    rest, faces, markers = pocket_mesh(p, spacing=req["spacing"], mouth_height=req["mouth_height"])
    metric_rest=material_rest(rest,p.get("flat_rest_panels",False),p.get("seam_height_m",0.))
    initial_blade = np.array([start_x,0.,.004 if req["blade_profile"] == "beveled" else .006])
    assert triangle_penetration(rest,faces,initial_blade,p["blade_size_m"],req["blade_profile"]) == 0, "Tool intersects cloth at reset"
    cloth_edges = surface_edges(faces)
    (out / "mesh.json").write_text(json.dumps({**audit(rest, faces), "markers": markers}, indent=2))
    app = None
    report = {"status": "initializing", "full_task_success": False, "product_gate_passed": False, "diagnostic_only": True}
    np.random.seed(req["seed"])
    try:
        from isaaclab.app import AppLauncher

        kit_args = kit_arguments(out)
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", multi_gpu=False, kit_args=kit_args).app
        import carb
        import omni.usd
        import torch
        from pxr import PhysxSchema, UsdGeom, UsdLux, UsdPhysics
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from cloth_solver import install as install_cloth_solver
        install_cloth_solver(req,out)
        from native import box, cloth, linear_rig, rubber_volume, set_attr, sew_lip

        sim = SimulationContext(SimulationCfg(device="cuda:0", dt=1 / req["physics_hz"], render_interval=req["physics_hz"] // 20,
                                              use_fabric=False, physics=PhysxCfg(), log_dir=str(out)))
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        # All supports are visible static collision geometry. No cloth pinning.
        box(stage, "/World/Table", (0.7, 0.5, 0.02), (0.15, 0, -0.01))
        if req["clamp_tail"]:
            for index, y in enumerate((-0.063, 0.063)):
                box(stage, f"/World/Weight{index}", (0.04, 0.01, 0.02), (0.165, y, 0.013), dynamic=True, mass=0.1, color=(0.7, 0.2, 0.3))
        assert not req["clamp_tail"] or req["repeats"] == 1, "Weight reset support is not implemented; one episode only"
        parts = [(np.arange(len(rest)), rest, faces)]
        shared = []
        if req["reinforced"]:
            parts, shared = split_lip(rest, faces, markers, config["estimated_rubber_lip"]["width_m"])
        simulation_parts = []
        volume_faces = None
        for index, (ids, part_rest, part_faces) in enumerate(parts):
            material = p if index == 0 else {**p, **config["estimated_rubber_lip"]}
            path = "/World/Pocket" if index == 0 else "/World/Lip"
            color = (0.15, 0.4, 0.75) if index == 0 else (0.2, 0.2, 0.2)
            if index == 1 and req["lip_model"] == "volume":
                nodes, elements, middle = extrude_surface(part_rest, part_faces, material["shell_thickness_m"])
                surface, volume_faces = rubber_volume(stage, nodes, elements, material, req["iterations"], path)
            else:
                nodes, elements, middle = part_rest, part_faces, np.arange(len(part_rest))
                surface = cloth(stage, nodes, elements, material, req["iterations"], path, color)
            simulation_parts.append((nodes, elements, middle))
            if req["ccd"]:
                set_attr(surface.GetPrim(), "physxDeformableBody:enableSpeculativeCCD", True)
                if not (index == 1 and req["lip_model"] == "volume"):
                    set_attr(surface.GetPrim(), "physxDeformableBody:collisionPairUpdateFrequency", 8)
        if req["reinforced"]:
            sew_lip(stage, parts, shared, filter_mode=req["seam_filter"], volume_faces=volume_faces,
                    enabled=not req["detach_lip"], grouping=req["filter_grouping"])
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr("GPU")
                api.CreateSolverTypeAttr("TGS")
                api.CreateEnableCCDAttr(False)
        linear_rig(stage, p["blade_size_m"], req["blade_profile"], start_x)
        rig = Articulation(ArticulationCfg(prim_path="/World/Rig", spawn=None,
                           actuators={"stage": ImplicitActuatorCfg(joint_names_expr=["Slide"], effort_limit_sim=10, stiffness=2000, damping=30)}))
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1800)
        deform = DeformableObject(DeformableObjectCfg(prim_path="/World/Pocket", spawn=None))
        objects = [deform]
        if req["reinforced"]:
            objects.append(DeformableObject(DeformableObjectCfg(prim_path="/World/Lip", spawn=None)))
        audit_scene = {"deformables": [], "attachments": [], "filters": [], "joints": []}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            schemas = prim.GetAppliedSchemas()
            if "OmniPhysicsDeformableBodyAPI" in schemas:
                assert prim.GetAttribute("physxDeformableBody:selfCollision").Get()
                assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                audit_scene["deformables"].append(path)
            if "Attachment" in prim.GetTypeName():
                assert path == "/World/LipSewing", "Unexpected material attachment"
                sources = [str(prim.GetRelationship("omniphysics:src" + str(i)).GetTargets()[0]) for i in range(2)]
                assert sources == ["/World/Pocket", "/World/Lip"]
                audit_scene["attachments"].append({"path": path, "sources": sources})
            if "CollisionFilter" in prim.GetTypeName():
                assert path == "/World/SewnNeighbors"
                audit_scene["filters"].append(path)
            if prim.IsA(UsdPhysics.Joint):
                assert path in ("/World/Rig/Anchor", "/World/Rig/Slide")
                audit_scene["joints"].append(path)
        (out / "scene-audit.json").write_text(json.dumps(audit_scene, indent=2))
        stage.GetRootLayer().Export(str(out / "scene.usda"))
        sim.reset()
        from runtime_checks import verify_gpu
        verify_gpu(out, sim, deform)
        view = deform.root_view

        def array(value):
            value = getattr(value, "torch", value)
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy().copy()
            return value.numpy().copy() if hasattr(value, "numpy") else np.asarray(value).copy()

        for obj, (part_rest, part_faces, _) in zip(objects, simulation_parts):
            part_view = obj.root_view
            native_faces = array(part_view.get_simulation_element_indices())[0].reshape(-1, part_faces.shape[1])[:len(part_faces)]
            native_rest = array(part_view.get_rest_nodal_positions())[0].reshape(-1, 3)[:len(part_rest)]
            assert np.allclose(native_rest, material_rest(part_rest,p.get("flat_rest_panels",False),p.get("seam_height_m",0.)), atol=1e-7), "Native rest geometry differs"
            assert np.array_equal(np.sort(native_faces, axis=1), np.sort(part_faces, axis=1)), "Native topology differs"
        blade_id = rig.find_bodies("Blade")[0][0]
        report["status"] = "running"
        report["reset_audits"] = []
        dt = 1 / req["physics_hz"]
        records, all_positions, all_velocities, blade_positions = [], [], [], []
        part_positions = [[] for _ in objects]
        part_velocities = [[] for _ in objects]
        trace_stream = (out / "steps.jsonl").open("w", buffering=1)
        (out / "result.json").write_text(json.dumps(report, indent=2))
        for repeat in range(req["repeats"]):
            # Reset is explicit and complete for this diagnostic's cloth and tool.
            for obj, (part_rest, _, _) in zip(objects, simulation_parts):
                initial = torch.zeros((1, len(part_rest), 6), device="cuda:0")
                initial[0, :, :3] = torch.as_tensor(part_rest, device="cuda:0")
                obj.write_nodal_state_to_sim_index(initial)
                obj.reset()
            zero = torch.zeros((1, 1), device="cuda:0")
            rig.write_joint_state_to_sim(zero, zero)
            rig.reset()
            reset_audit = {"repeat": repeat, "max_position_error_m": 0.0, "max_speed_m_s": 0.0}
            for obj, (part_rest, _, _) in zip(objects, simulation_parts):
                current = array(obj.root_view.get_simulation_nodal_positions())[0].reshape(-1, 3)[:len(part_rest)]
                velocity = array(obj.root_view.get_simulation_nodal_velocities())[0].reshape(-1, 3)[:len(part_rest)]
                reset_audit["max_position_error_m"] = max(reset_audit["max_position_error_m"], float(np.abs(current - part_rest).max()))
                reset_audit["max_speed_m_s"] = max(reset_audit["max_speed_m_s"], float(np.linalg.norm(velocity, axis=1).max()))
            assert reset_audit["max_position_error_m"] < 1e-6 and reset_audit["max_speed_m_s"] < 1e-6, "Incomplete coupon state reset"
            report["reset_audits"].append(reset_audit)
            for step in range(round(req["seconds"] / dt)):
                t = step * dt
                phase = "settle"
                center = np.array([start_x, 0.0, .004 if req["blade_profile"] == "beveled" else .006])
                if req["mode"] in ("insert", "cycle") and t >= 1:
                    phase = "insert" if t < hold_start else "tool_hold"
                    center[0] -= min(max(t - 1, 0), travel_seconds) * p["blade_speed_m_s"]
                    if req["mode"] == "cycle" and t >= withdraw_start:
                        center[0] += min(t - withdraw_start, travel_seconds) * p["blade_speed_m_s"]
                        phase = "withdraw" if t < released_start else "released_coupon"
                rig.set_joint_position_target(torch.tensor([[center[0] - start_x]], device="cuda:0", dtype=torch.float32))
                rig.write_data_to_sim()
                sim.step(render=step % max(1, req["physics_hz"] // 20) == 0)
                rig.update(dt)
                positions = np.zeros_like(rest)
                velocities = np.zeros_like(rest)
                seam_positions = []
                min_native_z, max_native_speed = float("inf"), 0.0
                for part_index, (obj, (ids, _, _), (part_rest, _, middle)) in enumerate(zip(objects, parts, simulation_parts)):

                    obj.update(dt)
                    local_pos = array(obj.root_view.get_simulation_nodal_positions())[0].reshape(-1, 3)[:len(part_rest)]
                    local_vel = array(obj.root_view.get_simulation_nodal_velocities())[0].reshape(-1, 3)[:len(part_rest)]
                    part_positions[part_index].append(local_pos)
                    part_velocities[part_index].append(local_vel)
                    min_native_z = min(min_native_z, float(local_pos[:, 2].min()))
                    max_native_speed = max(max_native_speed, float(np.linalg.norm(local_vel, axis=1).max()))
                    local_pos, local_vel = local_pos[middle], local_vel[middle]
                    positions[ids], velocities[ids] = local_pos, local_vel
                    if len(shared):
                        seam_positions.append(local_pos[np.searchsorted(ids, shared)])
                seam_gap = float(np.linalg.norm(seam_positions[0] - seam_positions[1], axis=1).max()) if len(shared) else 0.0
                measured_blade = array(rig.data.body_link_pose_w)[0, blade_id]
                measured_center = measured_blade[:3]
                strain = edge_strain(metric_rest, positions, faces, edges=cloth_edges)
                mouth = positions[markers["mouth_upper"], 2] - positions[markers["mouth_lower"], 2]
                max_pen = node_penetration(positions, measured_center, p["blade_size_m"], req["blade_profile"])
                finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
                reason = None
                if not finite:
                    reason = "nonfinite_state"
                elif np.max(np.abs(strain)) > config["gate"]["max_abs_edge_strain"]:
                    reason = "excessive_strain"
                elif max_pen > config["gate"]["max_penetration_m"] or positions[:, 2].min() < -config["gate"]["max_penetration_m"]:
                    reason = "fabric_penetration"
                elif seam_gap > 0.001:
                    reason = "seam_separation"
                row = {"repeat": repeat, "step": step, "time_s": t + dt, "phase": phase,
                       "rig_command_m": center.tolist(), "measured_blade_pose_xyzw": measured_blade.tolist(), "robot_action": None,
                       "max_seam_gap_m": seam_gap,
                       "mouth_median_height_m": float(np.median(mouth)), "max_abs_edge_strain": float(np.max(np.abs(strain))),
                       "max_speed_m_s": max_native_speed, "max_blade_node_penetration_m": max_pen,
                       "min_cloth_z_m": min_native_z, "score": 0.0, "invalid_reason": reason}
                records.append(row)
                trace_stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step % req["physics_hz"] == 0:
                    (out / "heartbeat.json").write_text(json.dumps(row, indent=2))
                    print(json.dumps({"repeat": repeat, "step": step, "time_s": t, "blade_x_m": float(measured_center[0]), "max_strain": row["max_abs_edge_strain"]}), flush=True)
                all_positions.append(positions)
                all_velocities.append(velocities)
                blade_positions.append(measured_center)
                if not finite or np.max(np.abs(strain)) > 2 or positions[:, 2].min() < -0.05:
                    report["termination_reason"] = reason or "escaped_work_area"
                    break
            if report.get("termination_reason"):
                break
        trace_stream.close()
        native_states = {}
        for index, (part_rest, elements, middle) in enumerate(simulation_parts):
            native_states.update({f"positions_{index}": part_positions[index],
                                  f"velocities_{index}": part_velocities[index],
                                  f"rest_{index}": material_rest(part_rest,p.get("flat_rest_panels",False),p.get("seam_height_m",0.)), f"elements_{index}": elements,
                                  f"middle_{index}": middle})
        np.savez_compressed(out / "native-states.partial.npz", **native_states)
        (out / "native-states.partial.npz").replace(out / "native-states.npz")
        np.savez_compressed(out / "states.partial.npz", positions=all_positions, velocities=all_velocities, blade_positions=blade_positions, rest=metric_rest, initial=rest, faces=faces)
        (out / "states.partial.npz").replace(out / "states.npz")
        with (out / "steps.jsonl").open("w") as f:
            for row in records:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        report.update(status="completed", steps=len(records), termination_reason=report.get("termination_reason", "diagnostic_time_limit"),
                      max_abs_edge_strain=max(r["max_abs_edge_strain"] for r in records),
                      max_blade_node_penetration_m=max(r["max_blade_node_penetration_m"] for r in records),
                      invalid_steps=sum(r["invalid_reason"] is not None for r in records), final=records[-1],
                      limitations=["No hardware parameters measured.", "Force-limited linear bench stage, not a robot demonstration.",
                                   "A single pocket does not validate the second pocket, passive frame lock, stock YAM grasps, or released functional retention."],
                      lip_model=req["lip_model"], seam_filter=req["seam_filter"],
                      attachment_enabled=bool(req["reinforced"] and not req["detach_lip"]),
                      rubber_lip_included=req["reinforced"], tail_weights_included=req["clamp_tail"])
    except Exception:
        report.update(status="error", traceback=traceback.format_exc(), termination_reason="runtime_error")
        traceback.print_exc()
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report), flush=True)
        if app:
            app.close()
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
