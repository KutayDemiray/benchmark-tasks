"""Compare the installed standalone PhysX SDK against an unchanged native scene.

This initialized passive diagnostic cannot earn benchmark credit. Unsupported
features or mismatched native mass/material/drive properties reject the run.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def validate_source(request, output):
    """Require an accepted original-mechanism recording with identical settings."""
    name = request['passive_sdk_scene']
    assert name and Path(name).name == name and name not in ('.', '..')
    source = output/name
    original = json.loads((source/'request.json').read_text())
    assert original['experiment'] == 'passive_contact'
    assert original['passive_source'] == request['passive_source']
    assert json.loads((source/'result.json').read_text())['status'] == 'completed'
    assert json.loads((source/'native-initialization-check.json').read_text())['passed']
    assert not (source/'native-constraint-rejection.json').exists()
    from passive_contact import native_physics_errors
    assert not native_physics_errors((source/'kit.log').read_text())
    assert request['physics_hz'] >= original['physics_hz'] and request['physics_hz'] % original['physics_hz'] == 0
    for key in ('iterations', 'spacing', 'contact_updates', 'contact_subiterations', 'ccd', 'product_profile'):
        assert original[key] == request[key], 'SDK comparison must preserve '+key
    assert bool(original.get('passive_free_space')) == bool(request.get('passive_free_space')), 'SDK gravity control must match its source'
    for record in (original, request):
        assert record['passive_mode'] == 'articulated' and record.get('passive_solver', 'TGS') == 'TGS'
        for key, value in record.items():
            if key.startswith('passive_') and key not in ('passive_source','passive_mode','passive_solver','passive_native_pipeline','passive_sdk_scene','passive_free_space','passive_pvd_record','passive_cloth_correction','passive_match_kit_defaults','passive_stop_when_settled'):
                assert value is None or value is False, 'SDK control cannot combine '+key
    assert not request.get('passive_native_pipeline')
    assert not original.get('passive_sdk_scene'), 'SDK source must be an independent Kit recording'
    return source


def verify_device(out, native_buffer):
    """Verify this physics-only process on its assigned GPU without inventing RTX evidence."""
    import torch
    from gpu_coordination import require_reservation, scheduler_binding
    gpu = require_reservation()
    xml = subprocess.check_output(['nvidia-smi','-q','-x'], text=True, timeout=5)
    allocations=[]
    for device in ET.fromstring(xml).findall('gpu'):
        for process in device.findall('processes/process_info'):
            if process.findtext('pid') == str(os.getpid()):
                allocations.append(dict(uuid=device.findtext('uuid'),type=process.findtext('type'),
                    memory_mib=float(process.findtext('used_memory').split()[0])))
    observed=str(torch.cuda.get_device_properties(0).uuid)
    checks=dict(native_tensor_device=str(native_buffer.device)=='cuda:0',
        cuda_uuid=observed.removeprefix('GPU-').lower()==gpu['uuid'].removeprefix('GPU-').lower(),
        observed_compute=any(p['uuid']==gpu['uuid'] and 'C' in p['type'] for p in allocations),
        no_other_device_work=all(p['uuid']==gpu['uuid'] for p in allocations))
    report=dict(pid=os.getpid(),job_id=os.environ.get('ROBOCURVE_JOB_ID'),expected_index=gpu['index'],
        expected_uuid=gpu['uuid'],cuda_uuid=observed,physics_tensor_device=str(native_buffer.device),
        renderer_enabled=False,active_renderer_physical_indices=[],allocations=allocations,
        native_api='ovphysx TensorBinding.read',active_cuda_gpus='0',checks=checks,verified=all(checks.values()))
    if 'ROBOCURVE_JOB_ID' in os.environ:
        scheduler_binding()
        from scheduler import identity,write_json
        report['process_identity']=identity(os.getpid())
        write_json(Path(os.environ['ROBOCURVE_JOB_DIR'])/'device-verification.json',report)
    (out/'gpu-verification.json').write_text(json.dumps(report,indent=2))
    assert report['verified'], 'Standalone native GPU assignment mismatch'


def run(request, out):
    """Restore one source scene, check native equivalence and record every step."""
    source=validate_source(request,out.parent)
    report=dict(status='initializing',diagnostic_only=True,benchmark_eligible=False,full_task_success=False)
    engine=stage=None; bindings=[]; errors=[]
    try:
        from gpu_coordination import require_reservation
        require_reservation()
        from ovphysx import PhysX,PhysXConfig,enable_python_logging,set_log_level
        from ovphysx.types import TensorType,LogLevel
        import ovstage
        import torch
        logger=logging.getLogger('pocket_standalone_physx')
        logger.setLevel(logging.INFO)
        handler=logging.FileHandler(out/'kit.log')
        handler.setFormatter(logging.Formatter('[%(levelname)s] [omni.physx.sdk] %(message)s'))
        logger.addHandler(handler)
        class NativeErrors(logging.Handler):
            def emit(self, record):
                if record.levelno>=logging.ERROR:
                    errors.append(record.getMessage())
        logger.addHandler(NativeErrors())
        enable_python_logging('pocket_standalone_physx')
        set_log_level(LogLevel.INFO)
        for name in ('scene.usda','geometry.npz','effective-parameters.json','frame-names.json','frame-geometry.json'):
            shutil.copyfile(source/name,out/name)
        source_request=json.loads((source/'request.json').read_text())
        if request['physics_hz'] != source_request['physics_hz']:
            original_scene=(out/'scene.usda').read_text()
            pattern=r'(uint physxScene:timeStepsPerSecond = )'+str(source_request['physics_hz'])+r'\b'
            refined,count=re.subn(pattern,lambda match:match.group(1)+str(request['physics_hz']),original_scene)
            assert count==1, 'Cannot isolate the authored timestep declaration'
            (out/'scene.usda').write_text(refined)
            (out/'sdk-timestep-refinement.json').write_text(json.dumps(dict(
                source_physics_hz=source_request['physics_hz'],physics_hz=request['physics_hz'],
                only_scene_time_steps_per_second_changed=True,source_scene_sha256=hashlib.sha256(original_scene.encode()).hexdigest(),
                scene_sha256=hashlib.sha256(refined.encode()).hexdigest()),indent=2))
        if request.get('passive_cloth_correction') or request.get('passive_match_kit_defaults'):
            from pxr import Usd
            from passive_defaults import cloth_correction, match_kit_defaults
            edit_stage=Usd.Stage.Open(str(out/'scene.usda'))
            if request.get('passive_match_kit_defaults'):
                match_kit_defaults(edit_stage, out)
            else:
                cloth_correction(edit_stage, out, request['passive_cloth_correction'])
            edit_stage.GetRootLayer().Save()
            edit_stage=None
        proof=json.loads((source/'passive-initialization.json').read_text())
        proof.update(standalone_sdk_source=source.name,standalone_sdk_scene_sha256=hashlib.sha256((source/'scene.usda').read_bytes()).hexdigest(),
            native_step_interface='installed ovphysx.PhysX.step_sync',direct_gpu_explicitly_enabled=True,diagnostic_only=True)
        (out/'passive-initialization.json').write_text(json.dumps(proof,indent=2))
        dynamics=json.loads((out.parent/request['passive_source']/'frame-dynamics.json').read_text())
        material=json.loads((out/'effective-parameters.json').read_text())
        if (source/'initial-frame-state.npz').exists():
            shutil.copyfile(source/'initial-frame-state.npz',out/'initial-frame-state.npz')
        else:
            # Older passive recordings retain the exact held source chunk instead.
            original=out.parent/proof['source_full_run']
            original_hz=json.loads((original/'request.json').read_text())['physics_hz']
            index=(proof['source_control_step']+1)*(original_hz//20)-1
            held=original/'physics-chunks'/f'{index//original_hz:05d}.npz'
            held_hash=hashlib.sha256(held.read_bytes()).hexdigest()
            assert held_hash==proof['held_chunk_sha256'], 'Held state provenance mismatch'
            with np.load(held) as chunk:
                poses=chunk['frame_body_poses'][index%original_hz].copy()
                joints=chunk['frame_joint_positions'][index%original_hz].copy()
            if source_request.get('passive_free_space'):
                poses[:,2]+=.2
            np.savez(out/'initial-frame-state.npz',poses=poses,joints=joints)
            (out/'initial-frame-reconstruction.json').write_text(json.dumps(dict(
                source_chunk=str(held.relative_to(out.parent)),source_sha256=held_hash,
                physics_step=index,height_offset_m=.2 if source_request.get('passive_free_space') else 0.,
                method='Original passive initialization from archived held native state; no new state fitting'),indent=2))
        with np.load(out/'initial-frame-state.npz') as initial:
            poses,joints=initial['poses'].copy(),initial['joints'].copy()
        with np.load(out/'geometry.npz') as geometry:
            x0,rest,faces=(geometry[key].copy() for key in ('initial','rest','faces'))
        recording={}
        if request.get('passive_pvd_record'):
            (out/'omnipvd').mkdir()
            recording=dict(omnipvd_output_enabled=True, omnipvd_ovd_recording_directory=str(out/'omnipvd'))
        engine=PhysX(active_cuda_gpus='0',config=PhysXConfig(cooked_collider_cache_dir=str(out/'cooked-cache'),
            carbonite_overrides={'/physics/suppressReadback':True}, **recording))
        assert ovstage.population.available(), 'Installed OVStage population bridge unavailable'
        stage=ovstage.Stage('pocket-mop-passive-sdk')
        ovstage.population.open_usd(stage,str(out/'scene.usda'),ordinal=1,domains=ovstage.PopulationDomain.PHYSICS)
        stage.advance_write_floor(ordinal=1).wait()
        engine.attach_ovstage(stage,read_ordinal=1)
        engine.step_sync(1/request['physics_hz'])  # Match the source warmup duration, before state restoration.
        def bind(name,path='/World/Frame'):
            binding=engine.create_tensor_binding(pattern=path,tensor_type=getattr(TensorType,name),raise_if_empty=True)
            bindings.append(binding)
            return binding
        def read(binding, dtype=np.float32):
            value=np.empty(binding.shape,dtype=dtype)
            binding.read(value)
            return value
        root=bind('ARTICULATION_ROOT_POSE'); q=bind('ARTICULATION_DOF_POSITION')
        links=bind('ARTICULATION_LINK_POSE'); velocity=bind('ARTICULATION_LINK_VELOCITY')
        body_names=links.body_names; joint_names=q.dof_names
        assert set(body_names)==set(dynamics['body_names']) and set(joint_names)==set(dynamics['joint_names'])
        bi=[body_names.index(name) for name in dynamics['body_names']]
        ji=[joint_names.index(name) for name in dynamics['joint_names']]
        order=[dynamics['joint_names'].index(name) for name in joint_names]
        native=dict(body_names=list(dynamics['body_names']),joint_names=list(dynamics['joint_names']),
            native_body_order=body_names,native_joint_order=joint_names,is_fixed_base=links.is_fixed_base)
        assert not links.is_fixed_base, 'Standalone frame must have a free floating base'
        disabled=read(bind('ARTICULATION_BODY_DISABLE_GRAVITY'),np.uint8)[:,bi]
        native['body_gravity_disabled']=disabled.tolist()
        assert not disabled.any(), 'Standalone frame gravity was disabled'
        inverse_mass=read(bind('ARTICULATION_BODY_INV_MASS'))[:,bi]
        native['inverse_masses']=inverse_mass.tolist()
        assert np.allclose(inverse_mass,1/np.asarray(dynamics['masses_kg']),rtol=1e-5)
        for kind in ('ARTICULATION_SHAPE_FRICTION_AND_RESTITUTION','ARTICULATION_CONTACT_OFFSET','ARTICULATION_REST_OFFSET'):
            native[kind.lower()]=read(bind(kind)).tolist()
        expected_shapes=sum(shape['enabled'] for shape in json.loads((out/'frame-geometry.json').read_text())['colliders'])
        assert np.shape(native['articulation_contact_offset'])==(1,expected_shapes), 'Native collision shape count changed'
        assert np.allclose(native['articulation_contact_offset'],.0005,atol=1e-9,rtol=0.)
        assert np.allclose(native['articulation_rest_offset'],0.,atol=1e-9,rtol=0.)
        assert np.allclose(native['articulation_shape_friction_and_restitution'],[.55,.55,0.],atol=1e-7,rtol=0.)
        native['joint_friction_properties']=read(bind('ARTICULATION_DOF_FRICTION_PROPERTIES'))[:,ji].tolist()
        native['maximum_joint_forces']=read(bind('ARTICULATION_DOF_MAX_FORCE'))[:,ji].tolist()
        joint_velocity_limits=bind('ARTICULATION_DOF_MAX_VELOCITY')
        if request.get('passive_match_kit_defaults'):
            from passive_defaults import match_kit_joint_velocity_defaults
            match_kit_joint_velocity_defaults(out, joint_velocity_limits, read)
        native['maximum_joint_velocities_rad_s']=read(joint_velocity_limits)[:,ji].tolist()
        assert abs(native['maximum_joint_forces'][0][-1]-10.)<1e-6

        comparisons=(('ARTICULATION_BODY_MASS','masses_kg',bi,1e-7),
            ('ARTICULATION_BODY_COM_POSE','local_com_poses_xyzw',bi,1e-6),
            ('ARTICULATION_BODY_INERTIA','inertias_kg_m2',bi,1e-8),
            ('ARTICULATION_DOF_STIFFNESS','joint_stiffnesses',ji,1e-7),
            ('ARTICULATION_DOF_DAMPING','joint_dampings',ji,1e-9))
        for kind,key,indices,tolerance in comparisons:
            value=read(bind(kind))[:,indices]
            native[key]=value.tolist()
            (out/'sdk-native-properties.json').write_text(json.dumps(native,indent=2))
            assert np.allclose(value,dynamics[key],atol=tolerance,rtol=1e-5), 'Native '+key+' changed'
        limits=read(bind('ARTICULATION_DOF_LIMIT'))[:,ji]
        native['joint_limits_rad']=limits.tolist()
        assert np.allclose(np.rad2deg(limits[0]),[[-90,0],[0,90],[-85,85],[-30,0],[-85,85],[-15,15]],atol=1e-4)
        properties=(('DEFORMABLE_MATERIAL_YOUNGS_MODULUS','young_modulus_pa'),
            ('DEFORMABLE_MATERIAL_POISSONS_RATIO','poisson_ratio'),
            ('DEFORMABLE_MATERIAL_DYNAMIC_FRICTION','dynamic_friction'),
            ('DEFORMABLE_MATERIAL_THICKNESS','shell_thickness_m'),
            ('DEFORMABLE_MATERIAL_BENDING_STIFFNESS','bend_modulus_pa'),
            ('DEFORMABLE_MATERIAL_ELASTICITY_DAMPING','elasticity_damping'),
            ('DEFORMABLE_MATERIAL_BENDING_DAMPING','bend_damping'))
        native['material']={}
        for kind,key in properties:
            value=read(bind(kind,'/World/Pocket/Fabric'))
            native['material'][key]=value.tolist()
            (out/'sdk-native-properties.json').write_text(json.dumps(native,indent=2))
            assert np.allclose(value,material[key],atol=1e-8,rtol=1e-5), 'Native material '+key+' changed'
        x=bind('SURFACE_DEFORMABLE_SIM_POSITION','/World/Pocket')
        v=bind('SURFACE_DEFORMABLE_SIM_VELOCITY','/World/Pocket')
        (out/'sdk-cloth-layout.json').write_text(json.dumps(dict(position_shape=x.shape,authored_nodes=len(x0)),indent=2))
        assert x.shape[0]==1 and x.shape[2]==3 and x.shape[1]>=len(x0), 'Unexpected native cloth shape: '+str(x.shape)
        native_rest=read(bind('SURFACE_DEFORMABLE_REST_POSITION','/World/Pocket'))
        native_faces=read(bind('SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES','/World/Pocket'),np.int32)
        layout=dict(position_shape=x.shape,rest_shape=native_rest.shape,face_shape=native_faces.shape,
            authored_nodes=len(rest),authored_faces=len(faces))
        (out/'sdk-cloth-layout.json').write_text(json.dumps(layout,indent=2))
        assert np.allclose(native_rest[0,:len(rest)],rest,atol=1e-7,rtol=0.)
        assert np.array_equal(native_faces[0,:len(faces)],faces), 'Native cloth node/element order changed'
        assert not native_rest[0,len(rest):].any(), 'Unexplained extra native rest nodes'
        tail=native_faces[0,len(faces):]
        assert np.all((tail==0)|(tail==-1)), 'Unexplained extra native elements'
        layout['padding_verified']=True
        (out/'sdk-cloth-layout.json').write_text(json.dumps(layout,indent=2))
        root.write(np.asarray(poses[dynamics['body_names'].index('Bridge')][None],dtype=np.float32))
        q.write(np.asarray(joints[order][None],dtype=np.float32))
        for kind in ('ARTICULATION_ROOT_VELOCITY','ARTICULATION_DOF_VELOCITY'):
            binding=bind(kind);binding.write(np.zeros(binding.shape,dtype=np.float32))
        initialized=read(x);initialized[0,:len(x0)]=x0
        x.write(initialized);v.write(np.zeros(v.shape,dtype=np.float32))
        engine.update_articulations_kinematic()
        restored=read(links)[0,bi]
        pose_error=float(np.max(np.linalg.norm(restored[:,:3]-poses[:,:3],axis=1)))
        angle_error=float(np.max((Rotation.from_quat(restored[:,3:]).inv()*Rotation.from_quat(poses[:,3:])).magnitude()))
        cloth_error=float(abs(read(x)[0,:len(x0)]-x0).max())
        proof.update(initial_link_position_error_m=pose_error,initial_link_rotation_error_rad=angle_error,initial_cloth_error_m=cloth_error)
        (out/'passive-initialization.json').write_text(json.dumps(proof,indent=2))
        assert pose_error<1e-6 and angle_error<1e-5 and cloth_error<1e-7
        gpu_buffer=torch.empty(links.shape,dtype=torch.float32,device='cuda:0')
        links.read(gpu_buffer)
        verify_device(out,gpu_buffer)
        assert not errors, 'Standalone native initialization errors'
        (out/'native-initialization-check.json').write_text(json.dumps(dict(passed=True,native_errors=errors,standalone_sdk=True),indent=2))
        chunks=out/'physics-chunks';chunks.mkdir()
        hz=request['physics_hz'];dt=1/hz;buffer=[];rows=[]
        from passive_settling import PassiveHold
        hold=PassiveHold(hz)
        previous_points=x0.astype(float)
        scene_audit=json.loads((out.parent/proof['source_full_run']/'scene-audit.json').read_text())
        table_top=scene_audit['table_center_m'][2]+scene_audit['table_size_m'][2]/2
        termination='diagnostic_time_limit'
        with (out/'steps.jsonl').open('w',buffering=1) as log:
            for step in range(round(request['seconds']*hz)):
                engine.step_sync(dt)
                actual=read(links)[0,bi];joint=read(q)[0,ji];speed=read(v)[0,:len(x0)]
                sample=dict(frame_body_poses=actual,frame_joint_positions=joint,frame_body_velocities=read(velocity)[0,bi],positions=read(x)[0,:len(x0)],velocities=speed)
                assert all(np.isfinite(value).all() for value in sample.values())
                normal=Rotation.from_quat(actual[0,3:]).apply([0,0,1])
                row=dict(step=step,time_s=(step+1)*dt,head_inclination_deg=float(np.rad2deg(np.arccos(np.clip(normal[2],-1,1)))),
                    bridge_height_m=float(actual[0,2]),max_cloth_speed_m_s=float(np.linalg.norm(speed,axis=1).max()),
                    wing_gear_error_deg=float(abs(np.rad2deg(joint[0]+joint[1]))))
                assert row['wing_gear_error_deg']<1., 'Native gear is absent or inconsistent'
                points=sample['positions'].astype(float)
                difference_speed=float(np.linalg.norm(points-previous_points,axis=1).max()*hz)
                previous_points=points
                settled=hold.observe(row['max_cloth_speed_m_s'], difference_speed,
                    row['head_inclination_deg'], float(abs(np.rad2deg(joint[:2])).max()),
                    row['bridge_height_m']-table_top)
                row.update(max_cloth_difference_speed_m_s=difference_speed,
                    passive_hold_seconds=hold.contiguous_steps/hz)
                should_stop=bool(request.get('passive_stop_when_settled') and settled)
                rows.append(row);buffer.append(sample);log.write(json.dumps(row)+'\n')
                if (step+1)%hz==0 or step==round(request['seconds']*hz)-1 or should_stop:
                    np.savez(chunks/f'{step//hz:05d}.npz',**{key:np.asarray([s[key] for s in buffer]) for key in buffer[0]})
                    buffer.clear();print(json.dumps(row),flush=True)
                assert not errors, 'Standalone native stepping error'
                if should_stop:
                    termination='diagnostic_settled_hold'
                    break
        report.update(status='completed',termination_reason=termination,physics_steps=len(rows),
            first_passive_hold_control_step=None if hold.first_stop_step is None else hold.first_stop_step//(hz//20),
            maximum_wing_gear_error_deg=max(row['wing_gear_error_deg'] for row in rows),
            maximum_head_inclination_deg=max(row['head_inclination_deg'] for row in rows),final=rows[-1],
            limitations=['Cold initialized diagnostic, not an ordinary robot execution or qualified task.'])
    except Exception:
        report.update(status='error',traceback=traceback.format_exc(),native_errors=errors)
        print(report['traceback'],flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2))
        for binding in reversed(bindings):
            binding.destroy()
        if engine is not None:
            if stage is not None:
                engine.detach_ovstage();stage.destroy()
            engine.release()
