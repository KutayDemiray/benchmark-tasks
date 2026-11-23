"""Requalify the existing laboratory latch test through the installed SDK.

This worker consumes the unchanged authored rig from a qualified reference and
uses only the original force-limited drive targets. It is never a robot demo.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import traceback

import numpy as np


def validate_source(request, output):
    name=request.get('mechanism_sdk_source')
    assert name and Path(name).name==name and name not in ('.','..')
    assert request['experiment']=='mechanism'
    assert not any(request.get(k) for k in ('with_cover','with_pocket','reinforced','frame_contact_correction_speed','cloth_contact_correction_speed','bend_damping_rate'))
    gate=json.loads((output/'mechanism-gate.json').read_text())
    assert gate['passed'] and name in (gate['positive_run'],gate['negative_run'])
    source=output/name
    original=json.loads((source/'request.json').read_text())
    assert json.loads((source/'result.json').read_text())['status']=='completed'
    assert json.loads((source/'mechanism-audit.json').read_text())['rigid_checks_passed']
    keys=('physics_hz','iterations','seconds','snap_throat','press_overtravel','retention_extension',
          'disable_latch_contact','mechanism_unlock','pedal_overtravel','unlock_lift','retention_raise')
    for key in keys:
        assert original.get(key,0)==request.get(key,0), 'SDK mechanism must preserve '+key
    assert hashlib.sha256((source/'source/frame_native.py').read_bytes()).digest()==hashlib.sha256(Path(__file__).with_name('frame_native.py').read_bytes()).digest()
    return source


def run(request, out):
    source=validate_source(request,out.parent)
    report=dict(status='initializing',diagnostic_only=True,full_task_success=False,
                mechanism_gate_passed=False,backend='installed_ovphysx')
    engine=stage=None;bindings=[];errors=[]
    try:
        from gpu_coordination import require_reservation
        require_reservation()
        from ovphysx import PhysX,PhysXConfig,enable_python_logging,set_log_level
        from ovphysx.types import TensorType,LogLevel
        import ovstage
        import torch
        from pxr import Usd,Sdf
        from mechanism_trajectory import targets
        from standalone_contact import verify_device
        logger=logging.getLogger('pocket_sdk_mechanism');logger.setLevel(logging.INFO)
        handler=logging.FileHandler(out/'kit.log')
        handler.setFormatter(logging.Formatter('[%(levelname)s] [omni.physx.sdk] %(message)s'))
        logger.addHandler(handler)
        class NativeErrors(logging.Handler):
            def emit(self, record):
                if record.levelno>=logging.ERROR:errors.append(record.getMessage())
        logger.addHandler(NativeErrors());enable_python_logging('pocket_sdk_mechanism');set_log_level(LogLevel.INFO)
        for name in ('scene.usda','mechanism.json'):
            shutil.copyfile(source/name,out/name)
        metadata=json.loads((out/'mechanism.json').read_text())
        original_result=json.loads((source/'result.json').read_text())
        original_gains=json.loads((source/'native-joint-stiffness.json').read_text())
        with np.load(source/'states.npz') as states:
            original_bodies=states['body_names'].tolist();original_joints=states['joint_names'].tolist()
        assert original_result['joint_names']==original_joints
        roots=['/World/Frame']+(['/World/PedalPusher'] if request['mechanism_unlock'] else [])
        usd=Usd.Stage.Open(str(out/'scene.usda'));before=usd.GetRootLayer().ExportToString()
        metadata_before={};changes=[]
        for path in roots:
            prim=usd.GetPrimAtPath(path);assert prim
            metadata_before[path]=usd.GetRootLayer().GetPrimAtPath(path).GetInfo('apiSchemas')
            # Preserve authored list-op syntax and unknown codeless API names.
            schemas=prim.GetMetadata('apiSchemas').GetAppliedItems()
            if 'PhysxArticulationAPI' not in schemas:
                prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit([*schemas,'PhysxArticulationAPI']))
            for name,value in (('physxArticulation:sleepThreshold',.005),('physxArticulation:stabilizationThreshold',1.)):
                attr=prim.GetAttribute(name);assert not attr or not attr.HasAuthoredValueOpinion()
                prim.CreateAttribute(name,Sdf.ValueTypeNames.Float,custom=False).Set(value)
                changes.append(dict(path=path,attribute=name,value=value))
        after=usd.GetRootLayer().ExportToString()
        for change in changes:usd.GetPrimAtPath(change['path']).RemoveProperty(change['attribute'])
        for path,value in metadata_before.items():usd.GetPrimAtPath(path).SetMetadata('apiSchemas',value)
        assert usd.GetRootLayer().ExportToString()==before, 'Unexpected physical edits'
        usd.GetRootLayer().ImportFromString(after);usd.GetRootLayer().Save();usd=None
        provenance=dict(source_run=source.name,diagnostic_only=True,changes=changes,
            source_scene_sha256=hashlib.sha256((source/'scene.usda').read_bytes()).hexdigest(),
            scene_sha256=hashlib.sha256((out/'scene.usda').read_bytes()).hexdigest(),
            original_drive_trajectory_preserved=True,geometry_materials_gains_unchanged=True)
        (out/'sdk-source-preservation.json').write_text(json.dumps(provenance,indent=2))
        engine=PhysX(active_cuda_gpus='0',config=PhysXConfig(cooked_collider_cache_dir=str(out/'cooked-cache'),
            carbonite_overrides={'/physics/suppressReadback':True}))
        stage=ovstage.Stage('pocket-mop-mechanism-sdk')
        ovstage.population.open_usd(stage,str(out/'scene.usda'),ordinal=1,domains=ovstage.PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait();engine.attach_ovstage(stage,read_ordinal=1)
        hz=request['physics_hz'];dt=1/hz;engine.step_sync(dt)
        def bind(name,path='/World/Frame'):
            binding=engine.create_tensor_binding(pattern=path,tensor_type=getattr(TensorType,name),raise_if_empty=True)
            bindings.append(binding);return binding
        def read(binding,dtype=np.float32):
            value=np.empty(binding.shape,dtype=dtype);binding.read(value);return value
        q=bind('ARTICULATION_DOF_POSITION');links=bind('ARTICULATION_LINK_POSE')
        assert set(q.dof_names)==set(original_joints) and set(links.body_names)==set(original_bodies)
        names=q.dof_names;ids={name:i for i,name in enumerate(names)}
        ji=[names.index(name) for name in original_joints]
        bi=[links.body_names.index(name) for name in original_bodies]
        native=dict(body_names=original_bodies,joint_names=original_joints,is_fixed_base=links.is_fixed_base)
        assert links.is_fixed_base, 'The laboratory guide must remain fixed, unlike the free task frame'
        for kind,key in (('ARTICULATION_DOF_STIFFNESS','stiffness'),('ARTICULATION_DOF_DAMPING','damping')):
            value=read(bind(kind))[:,ji];native[key]=value.tolist()
            assert np.allclose(value,original_gains[key],rtol=1e-5,atol=1e-8), 'Original native drive gains changed'
        maximum=read(bind('ARTICULATION_DOF_MAX_FORCE'));native['maximum_forces']=maximum[:,ji].tolist()
        assert maximum[0,ids['LaboratoryPress']]==15 and maximum[0,ids['ClipFlexure']]==10
        mass=read(bind('ARTICULATION_BODY_MASS'))[0]
        native['masses_kg']=mass[bi].tolist()
        for name,value in metadata['body_masses_kg'].items():
            assert np.isclose(mass[links.body_names.index(name)],value,rtol=1e-5,atol=1e-7)
        assert np.isclose(mass[links.body_names.index('LaboratoryBase')],1.)
        native['joint_limits_rad_or_m']=read(bind('ARTICULATION_DOF_LIMIT'))[:,ji].tolist()
        native['body_gravity_disabled']=read(bind('ARTICULATION_BODY_DISABLE_GRAVITY'),np.uint8)[:,bi].tolist()
        assert not np.asarray(native['body_gravity_disabled']).any()
        for kind in ('ARTICULATION_SHAPE_FRICTION_AND_RESTITUTION','ARTICULATION_CONTACT_OFFSET','ARTICULATION_REST_OFFSET'):
            native[kind.lower()]=read(bind(kind)).tolist()
        assert np.allclose(native['articulation_shape_friction_and_restitution'],[.55,.55,0.],atol=1e-7,rtol=0)
        assert np.allclose(native['articulation_contact_offset'],.0005,atol=1e-9,rtol=0)
        assert np.allclose(native['articulation_rest_offset'],0.,atol=1e-9,rtol=0)
        velocity_limit=bind('ARTICULATION_DOF_MAX_VELOCITY')
        max_vel=np.full(q.shape,5.939047009418473e36,dtype=np.float32)
        max_vel[0,ids['LaboratoryPress']]=np.finfo(np.float32).max
        velocity_limit.write(max_vel);assert np.array_equal(read(velocity_limit),max_vel)
        native['max_joint_velocities_rad_s_or_m_s']=read(velocity_limit)[:,ji].tolist()
        pusher_q=pusher_links=pusher_target=None;pusher_names=[]
        if request['mechanism_unlock']:
            path='/World/PedalPusher'
            pusher_q=bind('ARTICULATION_DOF_POSITION',path);pusher_links=bind('ARTICULATION_LINK_POSE',path)
            assert pusher_q.dof_names==['Press'] and pusher_links.is_fixed_base
            for kind,expected in (('ARTICULATION_DOF_STIFFNESS',800.),('ARTICULATION_DOF_DAMPING',10.),('ARTICULATION_DOF_MAX_FORCE',20.)):
                actual=read(bind(kind,path));assert np.allclose(actual,expected), 'Pusher drive changed'
                native['pusher_'+kind.lower()]=actual.tolist()
            pusher_q.write(np.zeros(pusher_q.shape,dtype=np.float32))
            b=bind('ARTICULATION_DOF_VELOCITY',path);b.write(np.zeros(b.shape,dtype=np.float32))
            pusher_target=bind('ARTICULATION_DOF_POSITION_TARGET',path);pusher_target.write(np.zeros(pusher_target.shape,dtype=np.float32))
            pusher_names=pusher_links.body_names
        initial_q=np.zeros(q.shape,dtype=np.float32)
        initial_q[0,ids['HingeLeft']]=-np.deg2rad(55.)
        initial_q[0,ids['HingeRight']]=np.deg2rad(55.)
        initial_q[0,ids['RockerHinge']]=-np.deg2rad(20.)
        q.write(initial_q)
        for kind in ('ARTICULATION_DOF_VELOCITY','ARTICULATION_DOF_VELOCITY_TARGET','ARTICULATION_DOF_ACTUATION_FORCE'):
            b=bind(kind);b.write(np.zeros(b.shape,dtype=np.float32))
        drive=bind('ARTICULATION_DOF_POSITION_TARGET');drive.write(np.zeros(drive.shape,dtype=np.float32))
        engine.update_articulations_kinematic()
        initial_poses=read(links)[0]
        assert np.allclose(initial_poses[links.body_names.index('Bridge'),:3],metadata['origin'],atol=1e-6,rtol=0)
        np.savez(out/'initial-native-state.npz',joint_positions=read(q)[:,ji],body_poses=initial_poses[bi])
        native['initial_joint_positions']=read(q)[:,ji].tolist()
        (out/'sdk-native-properties.json').write_text(json.dumps(native,indent=2))
        (out/'native-joint-stiffness.json').write_text(json.dumps({key:native[key] for key in ('stiffness','damping')},indent=2))
        gpu=torch.empty(links.shape,dtype=torch.float32,device='cuda:0');links.read(gpu);verify_device(out,gpu)
        assert not errors
        (out/'native-initialization-check.json').write_text(json.dumps(dict(passed=True,native_errors=errors),indent=2))
        rows=[];qs=[];poses=[];pushers=[]
        with (out/'steps.jsonl').open('w',buffering=1) as log:
            for step in range(round(request['seconds']*hz)):
                target,pedal,phase=targets(step*dt,request,metadata['origin'][2])
                commanded=np.zeros(q.shape,dtype=np.float32);commanded[0,ids['LaboratoryPress']]=target
                drive.write(commanded)
                if pusher_target is not None:pusher_target.write(np.array([[pedal]],dtype=np.float32))
                engine.step_sync(dt)
                joints=read(q)[0];bodies=read(links)[0,bi]
                assert np.isfinite(joints).all() and np.isfinite(bodies).all() and not errors
                wings=np.rad2deg(joints[[ids['HingeLeft'],ids['HingeRight']]])
                row=dict(step=step,time_s=(step+1)*dt,phase=phase,press_target_m=target,
                    joint_positions=joints[ji].tolist(),wing_angles_deg=wings.tolist(),
                    actuator_position_targets=commanded[0,ji].tolist(),
                    clip_deflection_deg=float(np.rad2deg(joints[ids['ClipFlexure']])),
                    rocker_angle_deg=float(np.rad2deg(joints[ids['RockerHinge']])),
                    gear_error_rad=float(abs(joints[ids['HingeLeft']]+joints[ids['HingeRight']])),
                    score=0.,robot_action=None,pedal_pusher_target_m=pedal,
                    pedal_pusher_joint_m=float(read(pusher_q)[0,0]) if pusher_q is not None else None)
                log.write(json.dumps(row)+'\n');rows.append(row);qs.append(joints[ji]);poses.append(bodies)
                if pusher_links is not None:pushers.append(read(pusher_links)[0])
                if step%hz==0:print(json.dumps(row),flush=True)
        np.savez_compressed(out/'states.npz',joint_positions=qs,body_poses=poses,
            joint_names=original_joints,body_names=original_bodies,pusher_poses=pushers,pusher_body_names=pusher_names)
        retained=[r for r in rows if r['phase']=='retention' and 10<=r['time_s']<=12+request.get('retention_extension',0.)]
        maximum_angle=max(abs(v) for r in retained for v in r['wing_angles_deg'])
        report.update(status='completed',termination_reason='diagnostic_time_limit',steps=len(rows),joint_names=original_joints,
            retention_window_seconds=[10,12+request.get('retention_extension',0.)],
            retained_window_max_wing_angle_deg=maximum_angle,last_two_seconds_max_wing_angle_deg=maximum_angle,
            passive_retention_observed=maximum_angle<3.,
            physical_unlock_observed=bool(pusher_q is not None and min(abs(v) for r in rows[-hz:] for v in r['wing_angles_deg'])>30.),
            limitations=['Original force-limited laboratory guide and pedal; no robot or cloth.',
                'Positive retention, negative control and independent rigid-contact audits are all required.'])
    except Exception:
        report.update(status='error',traceback=traceback.format_exc(),native_errors=errors)
        print(report['traceback'],flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2))
        for binding in reversed(bindings):binding.destroy()
        if engine is not None:
            if stage is not None:engine.detach_ovstage();stage.destroy()
            engine.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.request.read_text()),args.request.parent)
