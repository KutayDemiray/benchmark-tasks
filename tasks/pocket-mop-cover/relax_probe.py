"""Cold-start a recorded cloth state to diagnose native contact convergence.

This is not an episode replay or a grasp gate. Rest geometry and material remain
those of the source; positions and velocities are loaded once before stepping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",type=Path,required=True)
    request=p.parse_args().request
    req=json.loads(request.read_text())
    out=request.parent
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    source=out.parent/req["state_run"]
    assert source.resolve().parent==out.parent.resolve()
    original=json.loads((source/"request.json").read_text())
    assert original["product_profile"]=="base_micro" and original["experiment"]=="robot_grasp"
    geometry=np.load(source/"geometry.npz")
    rest,faces=geometry["rest"],geometry["faces"]
    last=sorted((source/"physics-chunks").glob("*.npz"))[-1]
    with np.load(last) as data:
        initial=data["positions"][-1].copy()
        velocity=data["velocities"][-1].copy()
    config=json.loads(Path(__file__).with_name("parameters.json").read_text())
    from robot_gate import fabric,array
    from mesh import edge_strain,surface_edges,material_rest
    material=json.loads((source/"effective-parameters.json").read_text())
    material["flat_rest_panels"]=bool(material.get("flat_rest_panels",False) or req.get("flat_rest_panels",False))
    material["flat_rest_bending"]=bool(material.get("flat_rest_bending",False) or req.get("flat_rest_bending",False))
    initial_shape=geometry["initial"].copy() if "initial" in geometry else rest.copy()
    rest=material_rest(rest,material["flat_rest_panels"],material.get("seam_height_m",0.))
    if req.get("bend_damping_rate") is not None:
        material["bend_damping"]=req["bend_damping_rate"]
    shift=np.array(req.get("diagnostic_origin_shift") or [0.,0.,0.])
    initial+=shift
    initial_shape+=shift
    rest+=shift
    material.update(collision_pair_updates=req["contact_updates"],
        collision_subiterations=req["contact_subiterations"],speculative_ccd=req["ccd"])
    (out/"input-state.json").write_text(json.dumps(dict(source_run=source.name,
        chunk=last.name,chunk_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
        material=material,coordinate_translation_m=shift.tolist(),limitation="Cold solver restart, not independent episode replay."),indent=2))
    app=None
    report=dict(status="initializing",diagnostic_only=True,full_task_success=False)
    try:
        from isaaclab.app import AppLauncher
        kit = kit_arguments(out)
        app=AppLauncher(headless=True,enable_cameras=True,device="cuda:0",multi_gpu=False,kit_args=kit).app
        import torch
        import omni.usd
        from pxr import PhysxSchema,UsdPhysics
        from isaaclab.sim import SimulationCfg,SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.assets import DeformableObject,DeformableObjectCfg
        from native import box,cloth
        from yam_arena import station
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
        box(stage,"/World/Table",station.TABLE_SIZE,np.asarray(station.TABLE_CENTER)+shift)
        scene=json.loads((source/"scene-audit.json").read_text())
        box(stage,"/World/Block",scene.get("support_size_m",[.25,.10,.12]),
            np.asarray(scene.get("support_center_m",[.30,.10,station.TABLE_TOP+.06]))+shift)
        cloth(stage,initial_shape,faces,material,req["iterations"])
        obj=DeformableObject(DeformableObjectCfg(prim_path="/World/Pocket",spawn=None))
        sim.reset()
        verify_gpu(out,sim,obj)
        state=np.concatenate((initial,velocity),axis=1)[None]
        obj.write_nodal_state_to_sim_index(torch.tensor(state,dtype=torch.float32,device="cuda:0"))
        obj.reset()
        stage.GetRootLayer().Export(str(out/"scene.usda"))
        np.savez_compressed(out/"geometry.npz",rest=rest,faces=faces)
        pos=[];vel=[];rows=[]
        edges=surface_edges(faces)
        with (out/"steps.jsonl").open("w",buffering=1) as log:
            for i in range(round(req["seconds"]*req["physics_hz"])):
                sim.step(render=False)
                obj.update(1/req["physics_hz"])
                x=array(obj.root_view.get_simulation_nodal_positions())[0].reshape(-1,3)[:len(rest)]
                v=array(obj.root_view.get_simulation_nodal_velocities())[0].reshape(-1,3)[:len(rest)]
                pos.append(x);vel.append(v)
                row=dict(step=i,time_s=(i+1)/req["physics_hz"],
                    max_speed_m_s=float(np.linalg.norm(v,axis=1).max()),
                    rms_speed_m_s=float(np.sqrt(np.mean(v*v))),
                    max_abs_edge_strain=float(abs(edge_strain(rest,x,faces,edges=edges)).max()),
                    score=0.)
                rows.append(row);log.write(json.dumps(row)+"\n")
                if (i+1)%req["physics_hz"]==0:
                    (out/"heartbeat.json").write_text(json.dumps(row,indent=2))
                    print(json.dumps(row),flush=True)
        np.savez_compressed(out/"states.npz",positions=pos,velocities=vel)
        speed=max(r["max_speed_m_s"] for r in rows[-2*req["physics_hz"]:])
        fd=float(np.linalg.norm(np.diff(np.array(pos[-2*req["physics_hz"]:]),axis=0),axis=2).max()*req["physics_hz"])
        report.update(status="completed",termination_reason="diagnostic_time_limit",
            final_native_speed_m_s=speed,final_difference_speed_m_s=fd,
            max_abs_edge_strain=max(r["max_abs_edge_strain"] for r in rows),
            released_speed_check=speed<=.03 and fd<=.03,
            product_gate_passed=False,scope="Contact convergence only; cold solver restart.")
    except Exception:
        report.update(status="error",traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        (out/"result.json").write_text(json.dumps(report,indent=2))
        if app:app.close()


if __name__=="__main__":
    main()
