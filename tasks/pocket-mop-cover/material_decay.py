"""Compare two constant bending rates on isolated free sheets without contact."""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from robot_gate import array,fabric


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    request=parser.parse_args().request
    req=json.loads(request.read_text());out=request.parent
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    report=dict(status='initializing',diagnostic_only=True,full_task_success=False,product_gate_passed=False)
    app=None
    try:
        from isaaclab.app import AppLauncher
        kit = kit_arguments(out)
        app=AppLauncher(headless=True,enable_cameras=True,device='cuda:0',multi_gpu=False,kit_args=kit).app
        import torch
        import omni.usd
        from pxr import PhysxSchema,UsdPhysics,Vt
        from isaaclab.sim import SimulationCfg,SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.assets import DeformableObject,DeformableObjectCfg
        from native import cloth,set_attr
        from runtime_checks import verify_gpu
        from mesh import surface_edges
        sim=SimulationContext(SimulationCfg(device='cuda:0',dt=1/req['physics_hz'],render_interval=req['physics_hz']//20,
            use_fabric=False,physics=PhysxCfg(),log_dir=str(out)))
        stage=omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api=PhysxSchema.PhysxSceneAPI.Apply(prim);api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr('GPU');api.CreateSolverTypeAttr('TGS')
        nx,ny=30,10;h=.001;length=nx*h;curvature=8*.001/length**2
        rest=np.array([[i*h-length/2,j*h-ny*h/2,.1] for i in range(nx+1) for j in range(ny+1)])
        initial=rest.copy();s=rest[:,0]
        initial[:,0]=np.sin(curvature*s)/curvature
        initial[:,2]+=(1-np.cos(curvature*s))/curvature
        faces=[]
        for i in range(nx):
            for j in range(ny):
                a=i*(ny+1)+j;b=a+ny+1;c=b+1;d=a+1
                faces.extend([[a,b,c],[a,c,d]])
        faces=np.array(faces,dtype=np.int32)
        edges=surface_edges(faces);rest_lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
        material=fabric(json.loads(Path(__file__).with_name('parameters.json').read_text()))
        objects=[];models=[]
        for i,rate in enumerate([.005,1.]):
            offset=np.array([0.,(-.025 if i==0 else .025),0.])
            parameters={**material,'bend_damping':rate}
            path=f'/World/Sample{i}'
            mesh=cloth(stage,initial+offset,faces,parameters,req['iterations'],path)
            set_attr(mesh.GetPrim(),'omniphysics:restShapePoints',Vt.Vec3fArray((rest+offset).tolist()))
            set_attr(mesh.GetPrim(),'physxDeformableBody:disableGravity',True)
            objects.append(DeformableObject(DeformableObjectCfg(prim_path=path,spawn=None)))
            models.append(dict(bend_damping_per_s=rate,offset_m=offset.tolist(),material=parameters))
        (out/'models.json').write_text(json.dumps(dict(models=models,
            scope='Zero-gravity free bending comparison; no clamps, contact, task scoring or product qualification.',
            initial_cylinder_curvature_per_m=curvature,nominal_end_to_midpoint_deflection_m=.001,
            reference='MATERIAL_MODEL.md; this is not the paper pendulum apparatus or a BaseMicro measurement.'),indent=2))
        sim.reset();verify_gpu(out,sim,objects[0])
        for obj,model in zip(objects,models):
            state=torch.zeros((1,len(rest),6),device='cuda:0')
            state[0,:,:3]=torch.as_tensor(initial+model['offset_m'],device='cuda:0')
            obj.write_nodal_state_to_sim_index(state);obj.reset()
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        all_positions=[[],[]];all_velocities=[[],[]];records=[]
        with (out/'steps.jsonl').open('w',buffering=1) as log:
            for step in range(round(req['seconds']*req['physics_hz'])):
                sim.step(render=False)
                row=dict(step=step,time_s=(step+1)/req['physics_hz'],samples=[])
                for i,obj in enumerate(objects):
                    obj.update(1/req['physics_hz'])
                    x=array(obj.root_view.get_simulation_nodal_positions())[0].reshape(-1,3)[:len(rest)]
                    v=array(obj.root_view.get_simulation_nodal_velocities())[0].reshape(-1,3)[:len(rest)]
                    centered=x-x.mean(0)
                    _,_,axes=np.linalg.svd(centered,full_matrices=False)
                    heights=centered@axes[-1]
                    row['samples'].append(dict(bend_damping_per_s=models[i]['bend_damping_per_s'],
                        rms_bending_displacement_m=float(np.sqrt(np.mean(heights**2))),
                        rms_internal_speed_m_s=float(np.sqrt(np.mean((v-v.mean(0))**2))),
                        max_abs_edge_strain=float(abs(np.linalg.norm(x[edges[:,1]]-x[edges[:,0]],axis=1)/rest_lengths-1).max()),
                        center_m=x.mean(0).tolist(),finite=bool(np.isfinite(x).all() and np.isfinite(v).all())))
                    all_positions[i].append(x);all_velocities[i].append(v)
                records.append(row);log.write(json.dumps(row)+'\n')
                if (step+1)%req['physics_hz']==0:
                    (out/'heartbeat.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
        np.savez_compressed(out/'states.npz',rest=rest,initial=initial,faces=faces,
            positions_0=all_positions[0],positions_1=all_positions[1],velocities_0=all_velocities[0],velocities_1=all_velocities[1])
        envelopes=[]
        for start in np.arange(0,req['seconds'],.5):
            selected=[r for r in records if start<=r['time_s']<start+.5]
            envelopes.append(dict(start_s=float(start),rms_bending_peak_m=[max(r['samples'][i]['rms_bending_displacement_m'] for r in selected) for i in range(2)]))
        report.update(status='completed',termination_reason='diagnostic_time_limit',steps=len(records),envelopes=envelopes,
            limitations=['Independent native rate comparison, not hardware calibration or a task gate.',
                'Gravity and contact deliberately absent for this material-only experiment.'])
    except Exception:
        report.update(status='error',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2))
        if app:app.close()


if __name__=='__main__':main()
