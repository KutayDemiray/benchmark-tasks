"""Repeat the original contact-only insertion/withdrawal coupon in the installed SDK."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np


def validate_source(request,output):
    name=request.get('probe_sdk_source');assert name and Path(name).name==name and name not in ('.','..')
    assert request['experiment']=='probe'
    source=output/name;gate=json.loads((output/'product-gate.json').read_text())
    assert gate['passed'] and gate['cycle_run']==name
    original=json.loads((source/'request.json').read_text())
    assert not any(request.get(key) for key in ('reinforced','clamp_tail','finite_seam','flat_rest_panels','flat_rest_bending','detach_lip','cloth_contact_correction_speed','bend_damping_rate'))
    for key in ('mode','physics_hz','iterations','spacing','seconds','repeats','contact_updates','contact_subiterations','ccd','product_profile','blade_profile','mouth_height'):
        assert request[key]==original[key], 'SDK coupon must preserve '+key
    assert request['repeats']==1
    assert json.loads((source/'physics-audit.json').read_text())['diagnostic_stability_passed']
    for name in ('native.py','mesh.py','parameters.json','blade_geometry.py'):
        assert (source/'source'/name).read_bytes()==Path(__file__).with_name(name).read_bytes(), 'Original physical source changed'
    return source


def run(request,out):
    source=validate_source(request,out.parent)
    report=dict(status='initializing',full_task_success=False,product_gate_passed=False,diagnostic_only=True,backend='installed_ovphysx')
    sim=None
    try:
        from sdk_session import SDKSession,author_defaults,surface_bindings
        from mesh import surface_edges,edge_strain
        from blade_geometry import node_penetration
        for name in ('scene.usda','scene-audit.json','mesh.json','effective-parameters.json'):
            shutil.copyfile(source/name,out/name)
        material=json.loads((out/'effective-parameters.json').read_text())['effective_probe']
        config=json.loads(Path(__file__).with_name('parameters.json').read_text())
        markers=json.loads((out/'mesh.json').read_text())['markers']
        with np.load(source/'states.npz') as recorded:
            initial,rest,faces=[recorded[key].copy() for key in ('initial','rest','faces')]
        author_defaults(out/'scene.usda',out,'/World/Pocket',['/World/Rig'])
        sim=SDKSession(out);hz=request['physics_hz'];dt=1/hz;sim.step(dt)
        q=sim.bind('ARTICULATION_DOF_POSITION','/World/Rig');links=sim.bind('ARTICULATION_LINK_POSE','/World/Rig')
        assert q.dof_names==['Slide'] and set(links.body_names)=={'Base','Blade'} and links.is_fixed_base
        native={}
        for kind,expected in (('ARTICULATION_DOF_STIFFNESS',2000.),('ARTICULATION_DOF_DAMPING',30.),('ARTICULATION_DOF_MAX_FORCE',10.),('ARTICULATION_BODY_MASS',.15)):
            value=sim.read(sim.bind(kind,'/World/Rig'));native[kind]=value.tolist()
            assert np.allclose(value,expected,atol=1e-7,rtol=1e-5), 'Original native rig property changed'
        limits=sim.read(sim.bind('ARTICULATION_DOF_LIMIT','/World/Rig'));native['joint_limits_m']=limits.tolist()
        assert np.allclose(limits,[[[-.205,.001]]],atol=1e-7,rtol=0)
        x,v=surface_bindings(sim,'/World/Pocket',material,rest,faces)
        q.write(np.zeros(q.shape,dtype=np.float32))
        for kind in ('ARTICULATION_DOF_VELOCITY','ARTICULATION_DOF_VELOCITY_TARGET','ARTICULATION_DOF_ACTUATION_FORCE'):
            binding=sim.bind(kind,'/World/Rig');binding.write(np.zeros(binding.shape,dtype=np.float32))
        drive=sim.bind('ARTICULATION_DOF_POSITION_TARGET','/World/Rig');drive.write(np.zeros(drive.shape,dtype=np.float32))
        points=sim.read(x);points[0,:len(initial)]=initial;x.write(points);v.write(np.zeros(v.shape,dtype=np.float32))
        sim.engine.update_articulations_kinematic()
        start_x=config['diagnostic_rig']['blade_initial_center_x_m'];end_x=config['diagnostic_rig']['blade_final_center_x_m']
        travel=(start_x-end_x)/material['blade_speed_m_s'];hold_start=1+travel;withdraw_start=3+travel;released_start=withdraw_start+travel
        blade_id=links.body_names.index('Blade')
        initial_blade=[start_x,0.,.004]
        assert np.allclose(sim.read(links)[0,blade_id,:3],initial_blade,atol=1e-6,rtol=0)
        reset=dict(repeat=0,max_position_error_m=float(abs(sim.read(x)[0,:len(initial)]-initial).max()),
            max_speed_m_s=float(np.linalg.norm(sim.read(v)[0,:len(initial)],axis=1).max()))
        assert reset['max_position_error_m']<1e-6 and reset['max_speed_m_s']==0
        (out/'sdk-native-properties.json').write_text(json.dumps(native,indent=2));sim.verify(links)
        report['reset_audits']=[reset]
        edges=surface_edges(faces);rows=[];positions=[];velocities=[];blade_positions=[]
        with (out/'steps.jsonl').open('w',buffering=1) as log:
            for step in range(round(request['seconds']*hz)):
                t=step*dt;phase='settle';center=np.asarray(initial_blade,dtype=float)
                if t>=1:
                    phase='insert' if t<hold_start else 'tool_hold'
                    center[0]-=min(max(t-1,0),travel)*material['blade_speed_m_s']
                    if t>=withdraw_start:
                        center[0]+=min(t-withdraw_start,travel)*material['blade_speed_m_s']
                        phase='withdraw' if t<released_start else 'released_coupon'
                drive.write(np.array([[center[0]-start_x]],dtype=np.float32));sim.step(dt)
                points=sim.read(x)[0,:len(initial)];speed=sim.read(v)[0,:len(initial)]
                assert np.isfinite(points).all() and np.isfinite(speed).all()
                blade=sim.read(links)[0,blade_id]
                strain=float(abs(edge_strain(rest,points,faces,edges=edges)).max())
                penetration=float(node_penetration(points,blade[:3],material['blade_size_m'],request['blade_profile']))
                minimum=float(points[:,2].min());reason=None
                if strain>config['gate']['max_abs_edge_strain']:reason='excessive_strain'
                elif penetration>config['gate']['max_penetration_m'] or minimum < -config['gate']['max_penetration_m']:reason='fabric_penetration'
                row=dict(repeat=0,step=step,time_s=t+dt,phase=phase,rig_command_m=center.tolist(),
                    measured_blade_pose_xyzw=blade.tolist(),robot_action=None,max_seam_gap_m=0.,
                    mouth_median_height_m=float(np.median(points[markers['mouth_upper'],2]-points[markers['mouth_lower'],2])),
                    max_abs_edge_strain=strain,max_speed_m_s=float(np.linalg.norm(speed,axis=1).max()),
                    max_blade_node_penetration_m=penetration,min_cloth_z_m=minimum,score=0.,invalid_reason=reason)
                log.write(json.dumps(row)+'\n');rows.append(row);positions.append(points);velocities.append(speed);blade_positions.append(blade[:3])
                if step%hz==0:print(json.dumps(row),flush=True)
                assert strain<2 and minimum>-.05, 'Catastrophic coupon distortion'
        np.savez_compressed(out/'states.npz',positions=positions,velocities=velocities,blade_positions=blade_positions,rest=rest,initial=initial,faces=faces)
        np.savez_compressed(out/'native-states.npz',positions_0=positions,velocities_0=velocities,rest_0=rest,elements_0=faces,middle_0=np.arange(len(rest)))
        (out/'sdk-source-preservation.json').write_text(json.dumps(dict(source_run=source.name,
            source_scene_sha256=hashlib.sha256((source/'scene.usda').read_bytes()).hexdigest(),
            original_tool_trajectory=True,original_geometry_material=True,
            state_writes_only_at_initialization=True,force_limited_slide_n=10.),indent=2))
        report.update(status='completed',steps=len(rows),termination_reason='diagnostic_time_limit',
            max_abs_edge_strain=max(r['max_abs_edge_strain'] for r in rows),
            max_blade_node_penetration_m=max(r['max_blade_node_penetration_m'] for r in rows),
            invalid_steps=sum(r['invalid_reason'] is not None for r in rows),final=rows[-1],
            lip_model='surface',seam_filter='element',attachment_enabled=False,rubber_lip_included=False,tail_weights_included=False,
            limitations=['Same contact-only single-pocket laboratory fixture; no robot or full-task score.',
                'Independent all-state contact/material and geometric seating audits are required.'])
    except Exception:
        report.update(status='error',termination_reason='runtime_error',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2))
        if sim is not None:sim.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.request.read_text()),args.request.parent)
