"""Dimensionally equivalent scene units for numerical-conditioning diagnostics."""

import numpy as np


def pose_to_metres(poses, scale):
    """Convert native poses to metres without changing quaternion components."""
    value = np.asarray(poses).copy()
    value[..., :3] /= scale
    return value


def velocity_to_metres(velocities, scale):
    """Convert spatial linear velocity to SI; angular velocity stays in rad/s."""
    value = np.asarray(velocities).copy()
    value[..., :3] /= scale
    return value


def scale_scene(stage, scale):
    """Apply one consistent length-unit conversion to this reduced scene."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt
    assert scale in (10., 100.)
    UsdGeom.SetStageMetersPerUnit(stage, 1./scale)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    powers = {
        'physics:gravityMagnitude': 1,
        'physics:localPos0': 1, 'physics:localPos1': 1,
        'physics:centerOfMass': 1, 'physics:diagonalInertia': 2,
        'physics:velocity': 1,
        'physxScene:bounceThreshold': 1, 'physxScene:frictionOffsetThreshold': 1,
        'physxScene:frictionCorrelationDistance': 1,
        'physxCollision:contactOffset': 1, 'physxCollision:restOffset': 1,
        'omniphysics:density': -3, 'omniphysics:youngsModulus': -1,
        'omniphysics:surfaceBendStiffness': -1, 'omniphysics:surfaceThickness': 1,
        'drive:angular:physics:stiffness': 2, 'drive:angular:physics:damping': 2,
        'drive:angular:physics:maxForce': 2,
    }
    changes = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            for attr in (api.GetMaxDepenetrationVelocityAttr(), api.GetMaxLinearVelocityAttr()):
                before = attr.Get(); assert attr.Set(before*scale)
                changes.append(dict(path=str(prim.GetPath()), attribute=attr.GetName(), power=1))
        for schema_name in ('PhysxRigidBodyAPI', 'PhysxArticulationAPI'):
            if prim.HasAPI(schema_name):
                prefix = 'physxRigidBody:' if schema_name == 'PhysxRigidBodyAPI' else 'physxArticulation:'
                for name in ('sleepThreshold', 'stabilizationThreshold'):
                    attr = prim.GetAttribute(prefix+name)
                    assert attr.Set(attr.Get()*scale**2)
                    changes.append(dict(path=str(prim.GetPath()), attribute=attr.GetName(), power=2))
        if prim.HasAPI('PhysxSurfaceDeformableBodyAPI'):
            for name in ('maxDepenetrationVelocity', 'maxLinearVelocity', 'settlingThreshold', 'sleepThreshold', 'selfCollisionFilterDistance'):
                attr = prim.GetAttribute('physxDeformableBody:'+name)
                before = attr.Get(); assert attr.Set(before*scale)
                changes.append(dict(path=str(prim.GetPath()), attribute=attr.GetName(), power=1))
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not attr.HasAuthoredValueOpinion():
                continue
            value = attr.Get()
            if name == 'xformOp:translate':
                assert attr.Set(type(value)(*(np.asarray(value, dtype=float)*scale).tolist()))
                power = 1
            elif name.startswith('xformOp:transform'):
                # No such op exists in the baseline. Reject rather than risk a
                # transform whose dimensionless rotation/scale is rescaled.
                raise AssertionError('Matrix xform needs explicit unit conversion: '+str(prim.GetPath()))
            elif name in ('points', 'velocities', 'omniphysics:restShapePoints'):
                assert attr.Set(Vt.Vec3fArray((np.asarray(value)*scale).tolist()))
                power = 1
            elif (prim.IsA(UsdGeom.Cube) and name == 'size') or (prim.IsA(UsdGeom.Capsule) and name in ('height', 'radius')):
                assert attr.Set(value*scale)
                power = 1
            elif name in powers:
                power = powers[name]
                factor = scale**power
                converted = type(value)(*(np.asarray(value, dtype=float)*factor).tolist()) if isinstance(value, (Gf.Vec3f, Gf.Vec3d)) else value*factor
                assert attr.Set(converted)
            else:
                continue
            changes.append(dict(path=str(prim.GetPath()), attribute=name, power=power))
    return dict(native_length_units_per_metre=scale, stage_metres_per_unit=1./scale,
        mass_units_kg=1., time_units_seconds=1., changes=changes,
        physical_parameters_preserved=True, outputs_converted_to_SI=True,
        scope='Equivalent units only; no material, geometry, physical gain, force or tolerance change.')


def preserve_recorded_mass_properties(stage, path, dynamics):
    """Author recorded SI inertias before changing units, avoiding recooking drift."""
    from pxr import Gf, UsdPhysics
    from scipy.spatial.transform import Rotation
    for i, name in enumerate(dynamics['body_names']):
        api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path+'/'+name))
        api.CreateMassAttr(dynamics['masses_kg'][0][i])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*dynamics['local_com_poses_xyzw'][0][i][:3]))
        inertia = np.array(dynamics['inertias_kg_m2'][0][i]).reshape(3, 3)
        values, axes = np.linalg.eigh((inertia+inertia.T)/2)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values.tolist()))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3].tolist())))
