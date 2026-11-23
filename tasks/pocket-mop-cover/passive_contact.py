"""Isolate recorded passive frame/cloth contact without robots or control actions.

The compound control freezes internal articulation motion before cooking while
preserving external geometry, total mass, center of mass and inertia. It is a
counterfactual diagnostic, never a benchmark implementation or qualification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gate import array


def check_request(request, output):
    """Reject diagnostic sources outside this task and accidental episode use."""
    source = request.get('passive_source')
    if request.get('experiment') != 'passive_contact':
        defaults = dict(passive_stop_when_settled=False, passive_match_kit_defaults=False, passive_cloth_correction=None, passive_pvd_record=False, diagnostic_output_root=None, passive_sdk_scene=None, passive_canonical_order=False, passive_source=None, passive_mode='articulated', passive_host_readback=False, passive_usd_readback=False, passive_native_pipeline=False, passive_handle_roll=None,
            passive_solver='TGS', passive_partitions=None, passive_external_forces_once=False, passive_no_self_contact=False,
            passive_contact_subiterations=None, passive_no_joint_friction=False,
            passive_deduplicate_wing_limits=False, passive_mimic_frequency=None, passive_free_space=False,
            passive_zero_bending=False, passive_center_body_frames=False, passive_position_iterations=None, passive_head_correction=False,
            passive_release_height=None, passive_no_contact_friction=False, passive_freeze_joints=None, passive_unit_scale=None, passive_no_frame_self_contact=False)
        assert all(request.get(key, value) == value for key, value in defaults.items()), 'Passive controls cannot enter ordinary episodes'
        return
    assert source and Path(source).name == source and source not in ('.', '..')
    assert not request.get('passive_pvd_record') or request['seconds'] <= .25, 'Native property captures must remain short'
    assert not request.get('passive_stop_when_settled') or (request.get('passive_sdk_scene') and request['seconds'] >= 2. and not request.get('passive_free_space') and not request.get('passive_pvd_record')), 'Settling termination needs a supported SDK contact recording of at least two seconds'
    assert not request.get('passive_match_kit_defaults') or (request.get('passive_sdk_scene') and not request.get('passive_cloth_correction'))
    assert not request.get('passive_cloth_correction') or (request['passive_mode']=='articulated' and not request.get('passive_unit_scale') and not request.get('passive_host_readback'))
    assert not request.get('passive_canonical_order') or (request['passive_mode']=='articulated' and not request.get('passive_sdk_scene') and not request.get('passive_freeze_joints') and not request.get('passive_center_body_frames') and not request.get('passive_unit_scale'))
    assert request.get('passive_handle_roll') is None or (request['passive_mode'] == 'articulated' and not request.get('passive_freeze_joints') and not request.get('passive_unit_scale') and not request.get('passive_center_body_frames'))
    assert not request.get('passive_native_pipeline') or (request['passive_mode'] == 'articulated' and not request.get('passive_host_readback') and not request.get('passive_unit_scale') and not request.get('passive_freeze_joints'))
    assert not request.get('passive_no_frame_self_contact') or request['passive_mode'] == 'articulated'
    assert request.get('passive_unit_scale') is None or (request['passive_mode'] == 'articulated' and not request.get('passive_center_body_frames') and not request.get('passive_freeze_joints'))
    assert not request.get('passive_usd_readback') or (request['passive_mode'] in ('maximal', 'hybrid') and request.get('passive_host_readback'))
    assert request['passive_mode'] in ('articulated', 'compound', 'bare', 'maximal', 'split_latch', 'reroot_handle', 'compound_articulation', 'hybrid')
    assert request.get('passive_freeze_joints') is None or (request['passive_mode'] == 'articulated' and not request.get('passive_deduplicate_wing_limits') and request.get('passive_mimic_frequency') is None)
    assert not request.get('passive_no_contact_friction') or request['passive_mode'] == 'articulated'
    assert request.get('passive_release_height') is None or (request['passive_mode'] == 'articulated' and not request.get('passive_free_space'))
    assert not request.get('passive_head_correction') or request['passive_mode'] == 'articulated'
    assert request.get('passive_position_iterations') is None or request['passive_mode'] == 'articulated'
    assert not request.get('passive_center_body_frames') or request['passive_mode'] == 'articulated'
    assert not request.get('passive_zero_bending') or request['passive_mode'] == 'articulated'
    assert not request.get('passive_free_space') or request['passive_mode'] == 'articulated'
    assert request.get('passive_mimic_frequency') is None or request['passive_mode'] == 'articulated'
    assert not request.get('passive_deduplicate_wing_limits') or request['passive_mode'] == 'articulated'
    assert not request.get('passive_no_joint_friction') or request['passive_mode'] == 'articulated'
    assert not request.get('passive_no_self_contact') or request['passive_mode'] == 'articulated'
    assert not request.get('passive_external_forces_once') or request.get('passive_solver', 'TGS') == 'TGS'
    assert request['passive_mode'] != 'hybrid' or request.get('passive_usd_readback')
    assert request['passive_mode'] not in ('maximal', 'hybrid') or request.get('passive_host_readback'), 'Gear constraints require host readback'
    assert not any(request.get(key) for key in ('robot_cover', 'release_probe_run', 'replay_run',
        'with_cover', 'with_pocket', 'reinforced', 'frame_contact_correction_speed',
        'cloth_contact_correction_speed', 'bend_damping_rate'))
    original = json.loads((output/source/'request.json').read_text())
    assert original.get('release_probe_hands_clear') and not original.get('release_probe_without_cover')
    assert not any(original.get(key) for key in ('release_probe_depenetration',
        'release_probe_deduplicate_colliders', 'release_probe_no_speculative_ccd', 'release_probe_contact_last'))
    assert original.get('release_probe_velocity_iterations') is None
    assert request['physics_hz'] >= original['physics_hz'] and request['physics_hz'] % original['physics_hz'] == 0
    for key in ('iterations', 'spacing', 'contact_updates', 'contact_subiterations', 'ccd', 'product_profile'):
        assert request[key] == original[key], f'Passive diagnostic must preserve {key}'
    assert 0 < request['seconds'] <= (20 if request.get('passive_sdk_scene') else 8) and request['repeats'] == 1
    if request.get('passive_sdk_scene'):
        from standalone_contact import validate_source
        validate_source(request, output)



def native_physics_errors(log):
    """Find native parser and solver errors, including rejected USD constraints."""
    return [line for line in log.splitlines() if 'PhysX error:' in line or
        ('[error]' in line.lower() and any(plugin in line for plugin in
            ('[omni.physx', '[omni.physics', '[isaaclab_physx')))]


def redundant_wing_limit_proof(left, right, coefficient, offset):
    """Require the gear to imply exactly the removed wing's angular interval."""
    implied = sorted(coefficient*np.asarray(left)+offset)
    assert np.allclose(implied, right, atol=1e-10, rtol=0.), 'Removing limits changes the feasible mechanism'
    return dict(retained_left_limits_deg=list(left), removed_right_limits_deg=list(right),
        gear_coefficient=coefficient, gear_offset_deg=offset, implied_right_limits_deg=implied,
        physical_feasible_angles_unchanged=True)

def compound_properties(poses, masses, local_coms, inertias, root_index=0):
    """Aggregate native body-frame inertias into the selected body's frame."""
    rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
    root_rotation = rotations[root_index]
    relative_rotations = np.einsum('ij,bjk->bik', root_rotation.T, rotations)
    relative_positions = (poses[:, :3]-poses[root_index, :3]) @ root_rotation
    centers = relative_positions + np.einsum('bij,bj->bi', relative_rotations, local_coms[:, :3])
    mass = masses.sum()
    center = (masses[:, None]*centers).sum(0)/mass
    inertia = np.zeros((3, 3))
    for m, c, rotation, body_inertia in zip(masses, centers, relative_rotations, inertias, strict=True):
        delta = c-center
        inertia += rotation @ body_inertia @ rotation.T + m*((delta @ delta)*np.eye(3)-np.outer(delta, delta))
    return float(mass), center, inertia, relative_positions, relative_rotations


def freeze_joint_group(stage, metadata, dynamics, poses, group):
    """Replace a selected joint group by fixed connections at its recorded pose."""
    from pxr import Gf, UsdPhysics
    names = (list(dynamics['joint_names']) if group == 'all' else
        ['HandleX', 'HandleY'] if group == 'handle' else ['HingeLeft', 'HingeRight', 'RockerHinge', 'ClipFlexure'])
    rotations = Rotation.from_quat(poses[:, 3:])
    for name in names:
        path = metadata['path']+'/'+name
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        parent = dynamics['body_names'].index(joint.GetBody0Rel().GetTargets()[0].name)
        child = dynamics['body_names'].index(joint.GetBody1Rel().GetTargets()[0].name)
        local0 = joint.GetLocalRot0Attr().Get()
        local0 = Rotation.from_quat([*local0.GetImaginary(), local0.GetReal()])
        local1 = (rotations[child].inv()*rotations[parent]*local0).as_quat()
        prim = joint.GetPrim()
        if prim.HasAPI('NewtonMimicAPI'):
            assert prim.RemoveAPI('NewtonMimicAPI')
        if prim.HasAPI(UsdPhysics.DriveAPI, 'angular'):
            assert prim.RemoveAPI(UsdPhysics.DriveAPI, 'angular')
        fixed = UsdPhysics.FixedJoint.Define(stage, path)
        fixed.CreateLocalRot1Attr(Gf.Quatf(float(local1[3]), Gf.Vec3f(*local1[:3])))
    return names


def rebase_body_origins(poses, offsets, *, inverse=False):
    """Express identical rigid bodies with translated local coordinate origins."""
    result = np.asarray(poses, dtype=float).copy()
    result[:, :3] += (-1. if inverse else 1.)*Rotation.from_quat(result[:, 3:]).apply(offsets)
    return result


def center_body_frames(stage, metadata, dynamics):
    """Move native body origins to the COM, preserving every physical transform."""
    from pxr import Gf, UsdGeom, UsdPhysics
    names = dynamics['body_names']
    offsets = np.array(dynamics['local_com_poses_xyzw'][0])[:, :3]
    for index, name in enumerate(names):
        center = offsets[index]
        prim = stage.GetPrimAtPath(metadata['path']+'/'+name)
        body_translate = prim.GetAttribute('xformOp:translate')
        assert body_translate.Set(Gf.Vec3d(*(np.asarray(body_translate.Get())+center)))
        for child in prim.GetChildren():
            assert child.IsA(UsdGeom.Gprim), 'Unexpected body child; cannot prove geometry equivalence'
            translate = child.GetAttribute('xformOp:translate')
            if translate:
                assert translate.Set(Gf.Vec3d(*(np.asarray(translate.Get())-center)))
            else:
                UsdGeom.Xformable(child).AddTranslateOp().Set(Gf.Vec3d(*(-center)))
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(dynamics['masses_kg'][0][index])
        api.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        inertia = np.array(dynamics['inertias_kg_m2'][0][index]).reshape(3, 3)
        values, axes = np.linalg.eigh((inertia+inertia.T)/2)
        assert values.min() > 0
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    for name in dynamics['joint_names']:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(metadata['path']+'/'+name))
        for side in (0, 1):
            body = getattr(joint, f'GetBody{side}Rel')().GetTargets()[0].name
            attr = getattr(joint, f'GetLocalPos{side}Attr')()
            assert attr.Set(Gf.Vec3f(*(np.asarray(attr.Get())-offsets[names.index(body)])))
    return offsets


def make_compound(stage, metadata, poses, dynamics):
    """Replace pre-cook joints by one equivalent compound body for diagnosis."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    path = metadata['path']
    root = stage.GetPrimAtPath(path)
    names = dynamics['body_names']
    masses = np.asarray(dynamics['masses_kg'][0])
    coms = np.asarray(dynamics['local_com_poses_xyzw'][0])
    inertias = np.asarray(dynamics['inertias_kg_m2'][0]).reshape(-1, 3, 3)
    mass, center, inertia, positions, rotations = compound_properties(poses, masses, coms, inertias, names.index('Bridge'))
    joints = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)]
    for joint in joints:
        stage.RemovePrim(joint)
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    for i, name in enumerate(names):
        prim = stage.GetPrimAtPath(path+'/'+name)
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAPI(UsdPhysics.MassAPI)
        prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*positions[i]))
        q = Rotation.from_matrix(rotations[i]).as_quat()
        transform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[3], Gf.Vec3d(*q[:3])))
    UsdPhysics.RigidBodyAPI.Apply(root)
    numerical = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    numerical.CreateSolverPositionIterationCountAttr(64)
    numerical.CreateSolverVelocityIterationCountAttr(4)
    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(mass)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    values, axes = np.linalg.eigh(inertia)
    assert min(values) > 0
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = Rotation.from_matrix(axes).as_quat()
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    return dict(mass_kg=mass, center_of_mass_m=center.tolist(), inertia_kg_m2=inertia.tolist(),
        link_positions_m=positions.tolist(), link_rotations=rotations.tolist(), removed_joints=joints,
        scope='Frozen internal degrees of freedom, same external geometry and aggregate mass properties; diagnostic only.')



def make_maximal(stage, metadata, poses, dynamics):
    """Use independent rigid bodies and native hinge/gear constraints for diagnosis."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    poses = np.asarray(poses, dtype=float)
    path = metadata['path']
    root = stage.GetPrimAtPath(path)
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    stage.GetPrimAtPath(path+'/HingeRight').RemoveAPI('NewtonMimicAPI')
    definitions = []
    for name in dynamics['joint_names']:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path+'/'+name))
        definitions.append(dict(name=name, parent=joint.GetBody0Rel().GetTargets()[0].name,
            child=joint.GetBody1Rel().GetTargets()[0].name, axis=joint.GetAxisAttr().Get()))
    for i, name in enumerate(dynamics['body_names']):
        prim = stage.GetPrimAtPath(path+'/'+name)
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*poses[i, :3]))
        q = poses[i, 3:]
        transform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[3], Gf.Vec3d(*q[:3])))
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(dynamics['masses_kg'][0][i])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*dynamics['local_com_poses_xyzw'][0][i][:3]))
        values, axes = np.linalg.eigh(np.array(dynamics['inertias_kg_m2'][0][i]).reshape(3, 3))
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
        numerical = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        numerical.CreateSolverPositionIterationCountAttr(64)
        numerical.CreateSolverVelocityIterationCountAttr(4)
    gear = PhysxSchema.PhysxPhysicsGearJoint.Define(stage, path+'/WingGears')
    gear.CreateBody0Rel().SetTargets([path+'/Left'])
    gear.CreateBody1Rel().SetTargets([path+'/Right'])
    gear.CreateHinge0Rel().SetTargets([path+'/HingeLeft'])
    gear.CreateHinge1Rel().SetTargets([path+'/HingeRight'])
    gear.CreateGearRatioAttr(1.)
    gear.CreateCollisionEnabledAttr(True)
    return definitions



def make_split_latch(stage, metadata, joint_names, *, reroot_handle=False):
    """Partition the same hinge-connected mechanism into two native articulations."""
    from pxr import PhysxSchema, UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg
    from frame_native import passive_actuators
    path = metadata['path']
    root = stage.GetPrimAtPath(path)
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    for name in (('Handle',) if reroot_handle else ('Bridge', 'Rocker')):
        prim = stage.GetPrimAtPath(path+'/'+name)
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        api.CreateEnabledSelfCollisionsAttr(True)
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(4)
    if not reroot_handle:
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path+'/RockerHinge')).CreateExcludeFromArticulationAttr(True)
    definitions = []
    for name in joint_names:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path+'/'+name))
        definitions.append(dict(name=name, parent=joint.GetBody0Rel().GetTargets()[0].name,
            child=joint.GetBody1Rel().GetTargets()[0].name, axis=joint.GetAxisAttr().Get()))
    if reroot_handle:
        parts = [Articulation(ArticulationCfg(prim_path=path+'/Handle', spawn=None, actuators=passive_actuators()))]
    else:
        parts = [Articulation(ArticulationCfg(prim_path=path+'/Bridge', spawn=None, actuators={})),
                 Articulation(ArticulationCfg(prim_path=path+'/Rocker', spawn=None, actuators=passive_actuators()))]
    return parts, definitions


def partition_readback(parts, names, method):
    """Collect each distinct native link into the original recorded body order."""
    values = {}
    for part in parts:
        data = array(getattr(part.root_view, method)())[0]
        for index, name in enumerate(part.body_names):
            assert name not in values
            values[name] = data[index]
    assert set(values) == set(names)
    return np.array([values[name] for name in names])

def measured_joint_angles(poses, names, definitions):
    """Measure signed single-axis hinge rotations from native body poses."""
    rotations = Rotation.from_quat(poses[:, 3:])
    result = []
    for joint in definitions:
        relative = rotations[names.index(joint['parent'])].inv()*rotations[names.index(joint['child'])]
        result.append(relative.as_rotvec()['XYZ'.index(joint['axis'])])
    return np.array(result)

def centered_handle_state(poses, joints, body_names, joint_names, roll_degrees):
    """Change only the gimbal roll in an initialized release-pose diagnostic."""
    poses, joints = np.array(poses, copy=True), np.array(joints, copy=True)
    bridge = poses[body_names.index('Bridge')]
    rotation = Rotation.from_quat(bridge[3:])
    x = np.deg2rad(roll_degrees)
    y = joints[joint_names.index('HandleY')]
    connector = poses[body_names.index('Connector')]
    connector[:3] = bridge[:3]+rotation.apply([.030, 0., .052])
    connector[3:] = (rotation*Rotation.from_euler('x', x)).as_quat()
    handle = poses[body_names.index('Handle')]
    handle[:3] = connector[:3]+Rotation.from_quat(connector[3:]).apply([0., 0., .010])
    handle[3:] = (Rotation.from_quat(connector[3:])*Rotation.from_euler('y', y)).as_quat()
    joints[joint_names.index('HandleX')] = x
    return poses, joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    path = parser.parse_args().request
    request = json.loads(path.read_text()); out = path.parent
    check_request(request, out.parent)
    from gpu_coordination import require_reservation, kit_arguments
    require_reservation()
    if request.get('passive_sdk_scene'):
        from standalone_contact import run
        return run(request, out)
    source = out.parent/request['passive_source']
    report = dict(status='initializing', diagnostic_only=True, benchmark_eligible=False, full_task_success=False)
    app = None
    try:
        original_manifest = json.loads((source/'manifest.json').read_text())['files']
        for name in ('frame_native.py', 'native.py', 'mesh.py', 'parameters.json'):
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == original_manifest[name]
        init = json.loads((source/'release-probe-initialization.json').read_text())
        recorded = out.parent/init['source_run']
        source_hz = json.loads((recorded/'request.json').read_text())['physics_hz']
        index = (init['source_control_step']+1)*(source_hz//20)-1
        chunk_path = recorded/'physics-chunks'/f'{index//source_hz:05d}.npz'
        with np.load(chunk_path) as chunk:
            poses = chunk['frame_body_poses'][index % source_hz].copy()
            joints = chunk['frame_joint_positions'][index % source_hz].copy()
        with np.load(source/'geometry.npz') as geometry:
            rest, initial, faces = (geometry[key].copy() for key in ('rest', 'initial', 'faces'))
        height = .2 if request.get('passive_free_space') else request.get('passive_release_height') or 0.
        if height:
            poses[:, 2] += height
            rest[:, 2] += height
            initial[:, 2] += height
        material = json.loads((source/'effective-parameters.json').read_text())
        if request.get('passive_zero_bending'):
            material['bend_modulus_pa'] = 0.
        if request.get('passive_no_contact_friction'):
            material['dynamic_friction'] = 0.
        if request.get('passive_contact_subiterations') is not None:
            assert request['passive_contact_subiterations'] == 1
            material['collision_subiterations'] = request['passive_contact_subiterations']
        dynamics = json.loads((source/'frame-dynamics.json').read_text())
        handle_pose_change = None
        if request.get('passive_handle_roll') is not None:
            previous = float(np.rad2deg(joints[dynamics['joint_names'].index('HandleX')]))
            poses, joints = centered_handle_state(poses, joints, dynamics['body_names'], dynamics['joint_names'], request['passive_handle_roll'])
            handle_pose_change = dict(previous_roll_deg=previous, initialized_roll_deg=request['passive_handle_roll'],
                only_gimbal_initial_pose_changed=True, task_demonstration=False)
            (out/'initial-handle-pose-change.json').write_text(json.dumps(handle_pose_change, indent=2))
        np.savez(out/'initial-frame-state.npz', poses=poses, joints=joints)
        from isaaclab.app import AppLauncher
        kit = kit_arguments(out)
        if request.get('passive_pvd_record'):
            (out/'omnipvd').mkdir()
            kit += f' --/persistent/physics/omniPvdOvdRecordingDirectory={out}/omnipvd/ --/physics/omniPvdOutputEnabled=true'
        app = AppLauncher(headless=True, enable_cameras=True, device='cuda:0', multi_gpu=False, kit_args=kit).app
        import torch
        import omni.usd
        from pxr import Sdf, Usd, UsdPhysics, Vt
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
        from frame_native import author_frame, passive_actuators
        from native import cloth, set_attr
        from runtime_checks import verify_gpu
        from passive_units import scale_scene, pose_to_metres, velocity_to_metres, preserve_recorded_mass_properties
        length_scale = request.get('passive_unit_scale') or 1.
        dt = 1/request['physics_hz']
        sim = SimulationContext(SimulationCfg(device='cuda:0', dt=dt, render_interval=request['physics_hz']//20,
            use_fabric=False, physics=PhysxCfg(), log_dir=str(out)))
        import carb
        if request.get('passive_host_readback'):
            carb.settings.get_settings().set('/physics/suppressReadback', False)
        if request.get('passive_usd_readback'):
            carb.settings.get_settings().set('/physics/updateToUsd', True)
            carb.settings.get_settings().set('/physics/updateVelocitiesToUsd', True)
        stage = omni.usd.get_context().get_stage()
        frozen = Usd.Stage.Open(str(source/'scene.usda'))
        for name in ('table', 'floor'):
            p = '/World/envs/env_0/'+name
            stage.DefinePrim('/World/envs/env_0', 'Xform')
            assert Sdf.CopySpec(frozen.GetRootLayer(), p, stage.GetRootLayer(), p)
        scene = stage.GetPrimAtPath('/physicsScene')
        source_scene = frozen.GetPrimAtPath('/physicsScene')
        for attr in source_scene.GetAttributes():
            if attr.HasAuthoredValueOpinion() and attr.GetName().startswith(('physics:', 'physxScene:')):
                scene.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        assert request.get('passive_solver', 'TGS') in ('PGS', 'TGS')
        scene.GetAttribute('physxScene:solverType').Set(request.get('passive_solver', 'TGS'))
        scene.GetAttribute('physxScene:timeStepsPerSecond').Set(request['physics_hz'])
        if request.get('passive_partitions') is not None:
            assert scene.GetAttribute('physxScene:gpuMaxNumPartitions').Set(request['passive_partitions'])
        if request.get('passive_free_space'):
            assert scene.GetAttribute('physics:gravityMagnitude').Set(0.)
        if request.get('passive_external_forces_once') or request.get('passive_solver', 'TGS') == 'PGS':
            scene.GetAttribute('physxScene:enableExternalForcesEveryIteration').Set(False)
        if frozen.GetPrimAtPath('/physicsScene/defaultMaterial'):
            assert Sdf.CopySpec(frozen.GetRootLayer(), '/physicsScene/defaultMaterial', stage.GetRootLayer(), '/physicsScene/defaultMaterial')
        metadata = author_frame(stage, origin=(0., 0., .004), snap_throat=True)
        if request.get('passive_canonical_order'):
            prim=stage.GetPrimAtPath(metadata['path'])
            original_order=[child.GetName() for child in prim.GetChildren()]
            before=stage.GetRootLayer().ExportToString()
            prim.SetChildrenReorder(sorted(original_order))
            after=stage.GetRootLayer().ExportToString()
            import re
            assert re.sub(r'^.*reorder nameChildren =.*\n','',after,flags=re.MULTILINE)==before
            (out/'link-order-comparison.json').write_text(json.dumps(dict(
                original_children=original_order,children=[child.GetName() for child in prim.GetChildren()],
                physical_attributes_identical=True,diagnostic_only=True),indent=2))
        if request.get('passive_head_correction'):
            from pxr import PhysxSchema
            changes = []
            for name in dynamics['body_names']:
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(metadata['path']+'/'+name))
                attr = api.GetMaxDepenetrationVelocityAttr()
                before = attr.Get()
                if name in ('Bridge', 'Left', 'Right'):
                    assert attr.Set(.1)
                else:
                    assert before == 3.
                changes.append(dict(body=name, before=before, after=attr.Get()))
            (out/'head-contact-correction.json').write_text(json.dumps(dict(
                changes=changes, set_only_before_cooking=True, physical_velocity_limit_changed=False,
                clip_and_gimbal_correction_unchanged=True, diagnostic_only=True,
                rationale='Separate heavy support-contact correction from the fast elastic latch; any candidate requires new mechanism/material gates and refinement.'), indent=2))
        if request.get('passive_no_frame_self_contact'):
            attr = stage.GetPrimAtPath(metadata['path']).GetAttribute('physxArticulation:enabledSelfCollisions')
            assert attr.Get() is True
            assert attr.Set(False)
        if request.get('passive_position_iterations') is not None:
            attr = stage.GetPrimAtPath(metadata['path']).GetAttribute('physxArticulation:solverPositionIterationCount')
            assert attr.Get() == 64
            assert attr.Set(request['passive_position_iterations'])
        if request.get('passive_deduplicate_wing_limits'):
            left = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(metadata['path']+'/HingeLeft'))
            right = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(metadata['path']+'/HingeRight'))
            prim = right.GetPrim()
            assert prim.GetAttribute('newton:mimicEnabled').Get() is True
            assert prim.GetRelationship('newton:mimicJoint').GetTargets() == [left.GetPath()]
            coefficient = prim.GetAttribute('newton:mimicCoef1').Get()
            offset = prim.GetAttribute('newton:mimicCoef0').Get()
            limit_proof = redundant_wing_limit_proof(
                [left.GetLowerLimitAttr().Get(), left.GetUpperLimitAttr().Get()],
                [right.GetLowerLimitAttr().Get(), right.GetUpperLimitAttr().Get()], coefficient, offset)
            # PhysX requires finite revolute limits to construct the mimic joint.
            # These inactive bounds strictly contain the gear-implied [0, 90].
            right.CreateLowerLimitAttr(-180.)
            right.CreateUpperLimitAttr(180.)
            limit_proof['inactive_right_limits_deg'] = [-180., 180.]
            (out/'redundant-limit-proof.json').write_text(json.dumps(limit_proof, indent=2))
        if request.get('passive_mimic_frequency') is not None:
            from pxr import PhysxSchema
            prim = stage.GetPrimAtPath(metadata['path']+'/HingeRight')
            assert prim.GetAttribute('newton:mimicCoef1').Get() == -1.
            assert prim.GetAttribute('newton:mimicCoef0').Get() == 0.
            assert prim.GetRelationship('newton:mimicJoint').GetTargets() == [Sdf.Path(metadata['path']+'/HingeLeft')]
            assert prim.RemoveAPI('NewtonMimicAPI')
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(prim, UsdPhysics.Tokens.rotY)
            mimic.CreateReferenceJointRel().SetTargets([metadata['path']+'/HingeLeft'])
            mimic.CreateReferenceJointAxisAttr(UsdPhysics.Tokens.rotY)
            mimic.CreateGearingAttr(1.)
            mimic.CreateOffsetAttr(0.)
            mimic.CreateNaturalFrequencyAttr(request['passive_mimic_frequency'])
            mimic.CreateDampingRatioAttr(1. if request['passive_mimic_frequency'] else 0.)
            (out/'mimic-compliance.json').write_text(json.dumps(dict(
                equation='HingeRight + HingeLeft = 0', natural_frequency_rad_s=request['passive_mimic_frequency'],
                damping_ratio=1. if request['passive_mimic_frequency'] else 0.,
                phase_dependent=False, fixed_before_cooking=True,
                native_schema='PhysxMimicJointAPI:rotY (installed compatibility schema)',
                limitation='Unqualified numerical constraint regularization; not measured gear elasticity. Requires convergence, gear residual and new mechanism qualification before candidate use.'), indent=2))
        if request.get('passive_no_joint_friction'):
            for name in ('HandleX', 'HandleY'):
                attr = stage.GetPrimAtPath(metadata['path']+'/'+name).GetAttribute('physxJoint:jointFriction')
                assert abs(attr.Get()-.03) < 1e-7
                assert attr.Set(0.)
        frozen_joints = []
        if request.get('passive_freeze_joints'):
            frozen_joints = freeze_joint_group(stage, metadata, dynamics, poses, request['passive_freeze_joints'])
        body_offsets = None
        if request.get('passive_center_body_frames'):
            body_offsets = center_body_frames(stage, metadata, dynamics)
            poses = rebase_body_origins(poses, body_offsets)
        def scaled_actuators():
            actuators = passive_actuators()
            for cfg in actuators.values():
                cfg.stiffness *= length_scale**2
                cfg.damping *= length_scale**2
                cfg.effort_limit_sim *= length_scale**2
            return actuators
        compound = None
        maximal = None
        split = None
        if request['passive_mode'] in ('compound', 'compound_articulation'):
            compound = make_compound(stage, metadata, poses, dynamics)
            if request['passive_mode'] == 'compound_articulation':
                from pxr import PhysxSchema
                prim = stage.GetPrimAtPath(metadata['path'])
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(64)
                api.CreateSolverVelocityIterationCountAttr(4)
                frame = Articulation(ArticulationCfg(prim_path=metadata['path'], spawn=None, actuators={},
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={})))
            else:
                frame = RigidObject(RigidObjectCfg(prim_path=metadata['path'], spawn=None))
        elif request['passive_mode'] in ('split_latch', 'reroot_handle'):
            split, definitions = make_split_latch(stage, metadata, dynamics['joint_names'],
                reroot_handle=request['passive_mode'] == 'reroot_handle')
            frame = split[0]
        elif request['passive_mode'] == 'hybrid':
            definitions = make_maximal(stage, metadata, poses, dynamics)
            from pxr import PhysxSchema
            for name in ('Bridge', 'Rocker'):
                prim = stage.GetPrimAtPath(metadata['path']+'/'+name)
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateEnabledSelfCollisionsAttr(True)
                api.CreateSolverPositionIterationCountAttr(64)
                api.CreateSolverVelocityIterationCountAttr(4)
            for name in ('HingeLeft', 'HingeRight', 'RockerHinge'):
                UsdPhysics.Joint(stage.GetPrimAtPath(metadata['path']+'/'+name)).CreateExcludeFromArticulationAttr(True)
            frame = None
        elif request['passive_mode'] == 'maximal':
            definitions = make_maximal(stage, metadata, poses, dynamics)
            maximal = []
            frame = None
        elif request.get('passive_native_pipeline'):
            frame = None
        else:
            frame = Articulation(ArticulationCfg(prim_path=metadata['path'], spawn=None,
                actuators={} if 'ClipFlexure' in frozen_joints else scaled_actuators(),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={})))
        cover = None
        if request['passive_mode'] != 'bare':
            mesh = cloth(stage, rest, faces, material, request['iterations'])
            set_attr(mesh.GetPrim(), 'omniphysics:restShapePoints', Vt.Vec3fArray(rest.tolist()))
            if request.get('passive_usd_readback'):
                mesh.CreatePointsAttr(Vt.Vec3fArray(initial.tolist()))
                mesh.CreateVelocitiesAttr(Vt.Vec3fArray(np.zeros_like(initial).tolist()))
            if request.get('passive_no_self_contact'):
                set_attr(mesh.GetPrim(), 'physxDeformableBody:selfCollision', False)
            if maximal is None and request['passive_mode'] != 'hybrid' and not request.get('passive_native_pipeline'):
                cover = DeformableObject(DeformableObjectCfg(prim_path='/World/Pocket', spawn=None))
        if request.get('passive_no_contact_friction'):
            friction_changes = []
            for prim in stage.Traverse():
                for name in ('physics:staticFriction', 'physics:dynamicFriction', 'omniphysics:dynamicFriction'):
                    attr = prim.GetAttribute(name)
                    if attr:
                        friction_changes.append(dict(path=str(prim.GetPath()), attribute=name, before=attr.Get()))
                        assert attr.Set(0.)
            assert len(friction_changes) >= 5
            (out/'friction-ablation.json').write_text(json.dumps(dict(
                counterfactual_only=True, all_contact_friction_zero=True, changes=friction_changes), indent=2))
        if length_scale != 1.:
            preserve_recorded_mass_properties(stage, metadata['path'], dynamics)
            unit_proof = scale_scene(stage, length_scale)
            unit_proof['recorded_mass_properties_authored_before_conversion'] = True
            (out/'unit-conversion.json').write_text(json.dumps(unit_proof, indent=2))
        if request.get('passive_cloth_correction'):
            from passive_defaults import cloth_correction
            cloth_correction(stage, out, request['passive_cloth_correction'])
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        if request.get('passive_usd_readback') or request.get('passive_native_pipeline'):
            # Host-mode tensor views do not need a DirectGPU warmup step. Avoid
            # advancing/deforming the authored cloth before its initial readback.
            from omni.physx import get_physx_interface, get_physx_simulation_interface
            from isaaclab.sim.utils.stage import get_current_stage_id
            assert get_physx_simulation_interface().attach_stage(get_current_stage_id())
            if request['passive_mode'] == 'hybrid':
                get_physx_interface().force_load_physics_from_usd()
            get_physx_interface().start_simulation()
            if request.get('passive_native_pipeline'):
                # DirectGPU requires a completed warmup before public tensor access.
                # Exact recorded poses and velocities are restored below.
                get_physx_simulation_interface().simulate(dt, 0.)
                get_physx_simulation_interface().fetch_results()
                import omni.physics.tensors as tensors
                from native_readback import ArticulationReadback, SurfaceReadback
                readback_view = tensors.create_simulation_view('torch')
                readback_view.set_subspace_roots('/')
                frame = ArticulationReadback(readback_view, metadata['path'])
                cover = SurfaceReadback(readback_view, '/World/Pocket')
                native_drives = dict(joint_names=frame.joint_names,
                    stiffnesses=array(frame.root_view.get_dof_stiffnesses())[0].tolist(),
                    dampings=array(frame.root_view.get_dof_dampings())[0].tolist(),
                    maximum_forces=array(frame.root_view.get_dof_max_forces())[0].tolist())
                (out/'native-drive-equivalence.json').write_text(json.dumps(native_drives, indent=2))
                order = [dynamics['joint_names'].index(name) for name in frame.joint_names]
                assert np.allclose(native_drives['stiffnesses'], np.array(dynamics['joint_stiffnesses'])[0, order], atol=1e-7)
                assert np.allclose(native_drives['dampings'], np.array(dynamics['joint_dampings'])[0, order], atol=1e-9)
                assert abs(native_drives['maximum_forces'][frame.joint_names.index('ClipFlexure')]-10.) < 1e-6
        else:
            sim.reset()
        if maximal is not None:
            import omni.physics.tensors as tensors
            from native_readback import RigidReadback, SurfaceReadback, UsdSurfaceReadback
            readback_view = tensors.create_simulation_view('torch')
            readback_view.set_subspace_roots('/')
            maximal = [RigidReadback(readback_view, metadata['path']+'/'+name) for name in dynamics['body_names']]
            frame = maximal[dynamics['body_names'].index('Bridge')]
            cover = (UsdSurfaceReadback(stage, '/World/Pocket') if request.get('passive_usd_readback')
                else SurfaceReadback(readback_view, '/World/Pocket'))
        if request['passive_mode'] == 'hybrid':
            import omni.physics.tensors as tensors
            from native_readback import ArticulationReadback, RigidLinkReadback, UsdSurfaceReadback
            readback_view = tensors.create_simulation_view('torch')
            readback_view.set_subspace_roots('/')
            split = [ArticulationReadback(readback_view, metadata['path']+'/'+name) for name in ('Bridge', 'Rocker')]
            split.extend(RigidLinkReadback(readback_view, metadata['path']+'/'+name) for name in ('Left', 'Right'))
            frame = split[0]
            cover = UsdSurfaceReadback(stage, '/World/Pocket')
            assert frame.joint_names == ['HandleX', 'HandleY']
            assert split[1].joint_names == ['ClipFlexure']
            native_drives = dict(joint_names=split[1].joint_names,
                stiffnesses=array(split[1].root_view.get_dof_stiffnesses())[0].tolist(),
                dampings=array(split[1].root_view.get_dof_dampings())[0].tolist(),
                gimbal_friction=array(frame.root_view.get_dof_friction_coefficients())[0].tolist())
            (out/'native-drive-equivalence.json').write_text(json.dumps(native_drives, indent=2))
            i = dynamics['joint_names'].index('ClipFlexure')
            assert np.allclose(native_drives['stiffnesses'], [dynamics['joint_stiffnesses'][0][i]], atol=1e-7)
            assert np.allclose(native_drives['dampings'], [dynamics['joint_dampings'][0][i]], atol=1e-9)
            assert np.allclose(native_drives['gimbal_friction'], [.03, .03], atol=1e-7)
        expected_suppression = not request.get('passive_host_readback', False)
        assert carb.settings.get_settings().get('/physics/suppressReadback') == expected_suppression
        native_errors = native_physics_errors((out/'kit.log').read_text())
        (out/'native-initialization-check.json').write_text(json.dumps(dict(
            passed=not native_errors, native_errors=native_errors, suppress_readback=expected_suppression,
            gpu_dynamics=scene.GetAttribute('physxScene:enableGPUDynamics').Get(),
            gravity_magnitude_m_s2=scene.GetAttribute('physics:gravityMagnitude').Get(),
            solver=scene.GetAttribute('physxScene:solverType').Get(),
            gpu_partitions=scene.GetAttribute('physxScene:gpuMaxNumPartitions').Get(),
            tensor_access=('public rigid tensor views; native USD surface export' if request.get('passive_usd_readback')
                else 'public native tensor views' if maximal is not None or request.get('passive_native_pipeline') else 'Isaac Lab assets'),
            external_forces_every_iteration=scene.GetAttribute('physxScene:enableExternalForcesEveryIteration').Get()), indent=2))
        assert not native_errors, 'Native physics rejected an authored constraint'
        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device='cuda:0')
        if body_offsets is not None:
            coms = array(frame.root_view.get_coms())[0]
            inertia = array(frame.root_view.get_inertias())[0]
            masses = array(frame.root_view.get_masses())[0]
            equivalence = dict(native_com_positions_m=coms[:, :3].tolist(),
                original_body_com_offsets_m=body_offsets.tolist(),
                max_inertia_error_kg_m2=float(abs(inertia-np.asarray(dynamics['inertias_kg_m2'][0])).max()),
                max_mass_error_kg=float(abs(masses-np.asarray(dynamics['masses_kg'][0])).max()),
                physical_geometry_and_joint_frames_preserved=True,
                output_body_poses_converted_to_original_frames=True)
            (out/'body-frame-equivalence.json').write_text(json.dumps(equivalence, indent=2))
            assert np.max(abs(coms[:, :3])) < 1e-7
            assert equivalence['max_inertia_error_kg_m2'] < 1e-8
            assert equivalence['max_mass_error_kg'] < 1e-7
        body_indices = joint_indices = None
        if compound is None and maximal is None and split is None:
            assert set(frame.body_names) == set(dynamics['body_names'])
            assert set(frame.joint_names) == set(dynamics['joint_names'])-set(frozen_joints)
            body_indices = [frame.body_names.index(name) for name in dynamics['body_names']]
            joint_indices = [dynamics['joint_names'].index(name) for name in frame.joint_names]
            root_name = frame.body_names[0]
        else:
            root_name = 'Bridge'
        if request.get('passive_canonical_order'):
            proof=json.loads((out/'link-order-comparison.json').read_text())
            proof.update(native_body_names=list(frame.body_names),native_joint_names=list(frame.joint_names))
            for method,key,tolerance in [('get_masses','masses_kg',1e-7),('get_coms','local_com_poses_xyzw',1e-6),('get_inertias','inertias_kg_m2',1e-8)]:
                values=array(getattr(frame.root_view,method)())[0,body_indices]
                assert np.allclose(values,dynamics[key][0],atol=tolerance,rtol=1e-5), 'Native '+key+' changed under reordering'
                proof[key]=values.tolist()
            (out/'link-order-comparison.json').write_text(json.dumps(proof,indent=2))
        bridge = poses[dynamics['body_names'].index(root_name)]
        native_bridge = bridge.copy(); native_bridge[:3] *= length_scale
        frame.write_root_link_pose_to_sim_index(root_pose=tensor(native_bridge[None]))
        frame.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1, 6))))
        if compound is None and maximal is None and split is None:
            if request.get('passive_deduplicate_wing_limits'):
                native_limits = np.rad2deg(array(frame.root_view.get_dof_limits())[0])
                measured_limits = {name: native_limits[frame.joint_names.index(name)].tolist()
                    for name in ('HingeLeft', 'HingeRight')}
                assert np.allclose(measured_limits['HingeLeft'], [-90., 0.], atol=1e-4)
                assert np.allclose(measured_limits['HingeRight'], [-180., 180.], atol=1e-4)
                limit_proof['native_limits_deg'] = measured_limits
                (out/'redundant-limit-proof.json').write_text(json.dumps(limit_proof, indent=2))
            if joint_indices:
                frame.write_joint_position_to_sim_index(position=tensor(joints[joint_indices][None]))
                frame.write_joint_velocity_to_sim_index(velocity=tensor(np.zeros_like(joints[joint_indices][None])))
        if split is not None:
            for part in split:
                pose = poses[dynamics['body_names'].index(part.body_names[0])]
                q = joints[[dynamics['joint_names'].index(name) for name in part.joint_names]].copy()
                part.write_root_link_pose_to_sim_index(root_pose=tensor(pose[None]))
                part.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1, 6))))
                part.write_joint_position_to_sim_index(position=tensor(q[None]))
                part.write_joint_velocity_to_sim_index(velocity=tensor(np.zeros_like(q[None])))
                array(part.data.body_link_pose_w)
        if maximal is not None:
            for body, pose in zip(maximal, poses, strict=True):
                body.write_root_link_pose_to_sim_index(root_pose=tensor(pose[None]))
                body.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1, 6))))
        if cover is not None:
            state = np.concatenate((initial*length_scale, np.zeros_like(initial)), axis=1)[None]
            cover.write_nodal_state_to_sim_index(tensor(state)); cover.reset()
            read_rest = array(cover.root_view.get_rest_nodal_positions())[0].reshape(-1, 3)[:len(rest)]/length_scale
            assert np.max(abs(read_rest-rest)) < 1e-7
        if not request.get('passive_native_pipeline') and request['passive_mode'] != 'hybrid':
            sim.forward()
        if cover is not None:
            actual_nodes = array(cover.root_view.get_simulation_nodal_positions())[0].reshape(-1, 3)[:len(initial)]/length_scale
            actual_velocity = array(cover.root_view.get_simulation_nodal_velocities())[0].reshape(-1, 3)[:len(initial)]/length_scale
            cloth_readback = dict(max_position_error_m=float(abs(actual_nodes-initial).max()),
                                  max_speed_m_s=float(np.linalg.norm(actual_velocity, axis=1).max()))
            (out/'initial-cloth-readback.json').write_text(json.dumps(cloth_readback, indent=2))
            assert cloth_readback['max_position_error_m'] < 1e-6 and cloth_readback['max_speed_m_s'] < 1e-7
        if compound is None and maximal is None and split is None:
            # Public body-data access refreshes FK even with render fabric disabled.
            restored = pose_to_metres(array(frame.data.body_link_pose_w)[0].reshape(-1, 7)[body_indices], length_scale)
            (out/'initial-link-readback.json').write_text(json.dumps(dict(
                expected=poses.tolist(), restored=restored.tolist(),
                maximum_translation_error_m=float(np.linalg.norm(restored[:, :3]-poses[:, :3], axis=1).max())), indent=2))
            assert np.max(np.linalg.norm(restored[:, :3]-poses[:, :3], axis=1)) < .0002
            assert np.max((Rotation.from_quat(restored[:, 3:])*Rotation.from_quat(poses[:, 3:]).inv()).magnitude()) < .002
        elif compound is not None:
            assert np.allclose(array(frame.root_view.get_masses()).ravel(), compound['mass_kg'], atol=1e-6)
            assert np.allclose(array(frame.root_view.get_coms()).reshape(-1, 7)[0, :3], compound['center_of_mass_m'], atol=1e-6)
            assert np.allclose(array(frame.root_view.get_inertias()).reshape(3, 3), compound['inertia_kg_m2'], atol=1e-7)
        if maximal is not None:
            restored = np.array([array(body.root_view.get_transforms())[0] for body in maximal])
            assert np.max(abs(restored-poses)) < 1e-6
            assert np.max(abs(measured_joint_angles(restored, dynamics['body_names'], definitions)-joints)) < 1e-5
            for i, body in enumerate(maximal):
                assert np.allclose(array(body.root_view.get_masses()).ravel(), dynamics['masses_kg'][0][i], atol=1e-7)
                assert np.allclose(array(body.root_view.get_inertias()).reshape(3, 3), np.array(dynamics['inertias_kg_m2'][0][i]).reshape(3, 3), atol=1e-8)
        if split is not None:
            restored = partition_readback(split, dynamics['body_names'], 'get_link_transforms')
            (out/'initial-partition-links.json').write_text(json.dumps(dict(
                expected=poses.tolist(), restored=restored.tolist(),
                groups=[dict(body_names=list(part.body_names), joint_names=list(part.joint_names),
                    positions=array(part.root_view.get_dof_positions()).tolist()) for part in split]), indent=2))
            assert np.max(np.linalg.norm(restored[:, :3]-poses[:, :3], axis=1)) < .0002
            assert np.max((Rotation.from_quat(restored[:, 3:])*Rotation.from_quat(poses[:, 3:]).inv()).magnitude()) < .002
            assert np.max(abs(measured_joint_angles(restored, dynamics['body_names'], definitions)-joints)) < .002
            assert np.allclose(partition_readback(split, dynamics['body_names'], 'get_masses'), dynamics['masses_kg'][0], atol=1e-7)
            if request['passive_mode'] == 'hybrid':
                assert np.allclose(partition_readback(split, dynamics['body_names'], 'get_inertias'), dynamics['inertias_kg_m2'][0], atol=1e-8)
                assert np.allclose(partition_readback(split, dynamics['body_names'], 'get_coms')[:, :3], np.array(dynamics['local_com_poses_xyzw'][0])[:, :3], atol=1e-7)
            (out/'partition-readback.json').write_text(json.dumps(dict(
                groups=[dict(body_names=list(part.body_names), joint_names=list(part.joint_names)) for part in split],
                maximum_initial_translation_error_m=float(np.linalg.norm(restored[:, :3]-poses[:, :3], axis=1).max()),
                external_hinges=([] if request['passive_mode'] == 'reroot_handle' else
                    ['HingeLeft', 'HingeRight', 'RockerHinge'] if request['passive_mode'] == 'hybrid' else ['RockerHinge']),
                native_gear=request['passive_mode'] == 'hybrid',
                root_changed_to_handle=request['passive_mode'] == 'reroot_handle', original_geometry_material_mass_and_hinge_limits=True,
                contact_filters_unchanged=True, clip_drive_unchanged=True), indent=2))
        if compound is None and maximal is None and split is None:
            measured_mass = array(frame.root_view.get_masses())[0][body_indices]
            measured_coms = pose_to_metres(array(frame.root_view.get_coms())[0][body_indices], length_scale)
            measured_inertia = array(frame.root_view.get_inertias())[0][body_indices]/length_scale**2
            # The installed GPU backend faults on zero-DOF floating mass-matrix
            # readback (R143d). Per-link native mass properties remain readable.
            matrix = None
            mass_check = dict(body_names=dynamics['body_names'], joint_names=list(frame.joint_names),
                masses_kg=measured_mass.tolist(), local_com_poses_xyzw=measured_coms.tolist(),
                inertias_kg_m2=measured_inertia.tolist(), generalized_mass_matrix=matrix,
                mass_matrix_not_queried='Native zero-DOF GPU API faults; preserve per-link evidence without calling it',
                total_mass_kg=float(measured_mass.sum()),
                max_mass_error_kg=float(abs(measured_mass-np.asarray(dynamics['masses_kg'][0])).max()),
                max_inertia_error_kg_m2=float(abs(measured_inertia-np.asarray(dynamics['inertias_kg_m2'][0])).max()))
            (out/'native-mass-response.json').write_text(json.dumps(mass_check, indent=2))
            assert mass_check['max_mass_error_kg'] < 1e-7 and mass_check['max_inertia_error_kg_m2'] < 1e-8
            if body_offsets is None:
                assert np.max(abs(measured_coms[:, :3]-np.asarray(dynamics['local_com_poses_xyzw'][0])[:, :3])) < 1e-7
        proof = dict(source_run=source.name, source_full_run=recorded.name, source_control_step=init['source_control_step'],
            mode=request['passive_mode'], source_physics_hz=source_hz, diagnostic_physics_hz=request['physics_hz'], robots_present=False, controllers_present=False, diagnostic_only=True,
            all_initial_velocities_zero=True, state_writes_only_before_recording=True, compound=compound,
            initial_handle_pose_change=handle_pose_change,
            native_usd_surface_readback=bool(request.get('passive_usd_readback')),
            hybrid_rigid_wings_with_articulated_gimbal_and_clip=request['passive_mode'] == 'hybrid',
            native_length_units_per_metre=length_scale,
            gpu_partitions=scene.GetAttribute('physxScene:gpuMaxNumPartitions').Get(),
            native_step_interface=('IPhysx.update_simulation' if request.get('passive_usd_readback') else
                'IPhysxSimulation.simulate + fetch_results' if request.get('passive_native_pipeline') else 'SimulationContext.step'),
            native_pipeline_without_framework_assets=bool(request.get('passive_native_pipeline')),
            collision_iteration_multiplier=material['collision_subiterations'],
            redundant_wing_limits_removed=bool(request.get('passive_deduplicate_wing_limits')),
            mimic_frequency_rad_s=request.get('passive_mimic_frequency'),
            free_space_counterfactual=bool(request.get('passive_free_space')),
            zero_bending_counterfactual=bool(request.get('passive_zero_bending')),
            native_body_frames_centered=body_offsets is not None,
            articulation_position_iterations=request.get('passive_position_iterations') or 64,
            head_only_contact_correction=bool(request.get('passive_head_correction')),
            initial_assembly_height_offset_m=height,
            contact_friction_disabled_counterfactual=bool(request.get('passive_no_contact_friction')),
            frozen_joints_counterfactual=frozen_joints,
            frame_self_contact_disabled_counterfactual=bool(request.get('passive_no_frame_self_contact')),
            joint_friction_disabled_counterfactual=bool(request.get('passive_no_joint_friction')),
            cloth_self_contact_disabled_counterfactual=bool(request.get('passive_no_self_contact')),
            maximal_joint_note=None if maximal is None else 'Native rigid hinges and 1:1 gear replace reduced-coordinate joints. Articulation-only joint friction is not supported by this diagnostic control; no benchmark qualification.',
            source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                ('scene.usda', 'geometry.npz', 'effective-parameters.json', 'frame-dynamics.json', 'release-probe-initialization.json')},
            held_chunk_sha256=hashlib.sha256(chunk_path.read_bytes()).hexdigest())
        (out/'passive-initialization.json').write_text(json.dumps(proof, indent=2))
        (out/'frame-geometry.json').write_text(json.dumps(metadata, indent=2))
        (out/'frame-names.json').write_text(json.dumps(dict(body_names=dynamics['body_names'], joint_names=dynamics['joint_names']), indent=2))
        (out/'effective-parameters.json').write_text(json.dumps(material, indent=2))
        np.savez_compressed(out/'geometry.npz', rest=rest, initial=initial, faces=faces)
        chunks = out/'physics-chunks'; chunks.mkdir()
        buffered = []; rows = []
        with (out/'steps.jsonl').open('w', buffering=1) as log:
            for step in range(round(request['seconds']/dt)):
                if request.get('passive_usd_readback'):
                    get_physx_interface().update_simulation(dt, step*dt)
                    get_physx_simulation_interface().fetch_results()
                elif request.get('passive_native_pipeline'):
                    get_physx_simulation_interface().simulate(dt, 0.)
                    get_physx_simulation_interface().fetch_results()
                else:
                    sim.step(render=False)
                frame.update(dt)
                if step == 0:
                    verify_gpu(out, sim, frame, expected_tensor_device=frame.device if maximal is not None or request['passive_mode'] == 'hybrid' else 'cuda:0')
                if compound is None and maximal is None and split is None:
                    actual = pose_to_metres(array(frame.root_view.get_link_transforms())[0].reshape(-1, 7)[body_indices], length_scale)
                    q = joints.copy()
                    if joint_indices:
                        q[joint_indices] = array(frame.root_view.get_dof_positions())[0]
                    v = velocity_to_metres(array(frame.root_view.get_link_velocities())[0].reshape(-1, 6)[body_indices], length_scale)
                elif split is not None:
                    for part in split:
                        part.update(dt)
                    actual = partition_readback(split, dynamics['body_names'], 'get_link_transforms')
                    q = measured_joint_angles(actual, dynamics['body_names'], definitions)
                    v = partition_readback(split, dynamics['body_names'], 'get_link_velocities')
                elif maximal is not None:
                    actual = np.array([array(body.root_view.get_transforms())[0] for body in maximal])
                    q = measured_joint_angles(actual, dynamics['body_names'], definitions)
                    v = np.array([array(body.root_view.get_velocities())[0] for body in maximal])
                else:
                    root = array(frame.root_view.get_root_transforms() if request['passive_mode'] == 'compound_articulation' else frame.root_view.get_transforms())[0]
                    rotation = Rotation.from_quat(root[3:])
                    actual = np.empty_like(poses)
                    actual[:, :3] = root[:3]+rotation.apply(compound['link_positions_m'])
                    actual[:, 3:] = (rotation*Rotation.from_matrix(compound['link_rotations'])).as_quat()
                    q = joints.copy()
                    v = array(frame.root_view.get_link_velocities() if request['passive_mode'] == 'compound_articulation' else frame.root_view.get_velocities()).reshape(-1, 6)
                if body_offsets is not None:
                    actual = rebase_body_origins(actual, body_offsets, inverse=True)
                sample = dict(frame_body_poses=actual, frame_joint_positions=q, frame_body_velocities=v)
                if cover is not None:
                    cover.update(dt)
                    x = array(cover.root_view.get_simulation_nodal_positions())[0].reshape(-1, 3)[:len(initial)]/length_scale
                    v = array(cover.root_view.get_simulation_nodal_velocities())[0].reshape(-1, 3)[:len(initial)]/length_scale
                    assert np.isfinite(x).all() and np.isfinite(v).all()
                    sample.update(positions=x, velocities=v)
                normal = Rotation.from_quat(actual[0, 3:]).apply([0., 0., 1.])
                row = dict(step=step, time_s=(step+1)*dt, head_inclination_deg=float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1)))),
                    bridge_height_m=float(actual[0, 2]), max_cloth_speed_m_s=None if cover is None else float(np.linalg.norm(v, axis=1).max()),
                    wing_gear_error_deg=float(abs(np.rad2deg(q[dynamics['joint_names'].index('HingeLeft')]+q[dynamics['joint_names'].index('HingeRight')]))))
                rows.append(row); log.write(json.dumps(row)+'\n'); buffered.append(sample)
                if (step+1) % request['physics_hz'] == 0 or step == round(request['seconds']/dt)-1:
                    np.savez(chunks/f'{step//request["physics_hz"]:05d}.npz', **{key: np.asarray([s[key] for s in buffered]) for key in buffered[0]})
                    buffered.clear()
                    (out/'heartbeat.json').write_text(json.dumps(row, indent=2)); print(json.dumps(row), flush=True)
        if request.get('passive_usd_readback'):
            usd_proof = dict(native_updates=cover.native_updates,
                maximum_observed_node_displacement_m=cover.maximum_displacement,
                maximum_observed_native_velocity_m_s=cover.maximum_speed,
                live_positions_verified=cover.maximum_displacement > 1e-6,
                live_velocities_verified=cover.maximum_speed > 1e-6)
            (out/'usd-surface-readback.json').write_text(json.dumps(usd_proof, indent=2))
            assert usd_proof['live_positions_verified'] and usd_proof['live_velocities_verified'], 'Native USD surface export is stale'
        assert not native_physics_errors((out/'kit.log').read_text()), 'Native physics error during recording'
        report.update(status='completed', termination_reason='diagnostic_time_limit', physics_steps=len(rows),
            maximum_wing_gear_error_deg=max(r['wing_gear_error_deg'] for r in rows),
            maximum_head_inclination_deg=max(r['head_inclination_deg'] for r in rows), final=rows[-1],
            limitations=['Cold contact cache and initialized installed geometry, not a from-reset task.',
                'The compound control removes physical degrees of freedom and is forbidden as a task model.'])
    except Exception:
        report.update(status='error', traceback=traceback.format_exc()); print(report['traceback'], flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(report, indent=2))
        if app:
            app.close()


if __name__ == '__main__':
    main()
