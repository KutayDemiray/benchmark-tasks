"""Native surface FEM coupons and a contact-only, passive spring socket.

No pairing IDs enter the mechanics. Permanent vertex attachments represent
hardware rivets only. The diagnostic fixture is explicitly separate.
"""

import math
import numpy as np


def set_attr(prim, name, value):
    attr = prim.GetAttribute(name)
    assert attr, f'Missing native schema property: {name}'
    assert attr.Set(value), name


def body(stage, path, pos, mass, fixed=False, rotation=0.):
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*pos))
    root.AddOrientOp().Set(Gf.Quatf(math.cos(rotation / 2), 0, math.sin(rotation / 2), 0))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateSolverPositionIterationCountAttr(64)
    api.CreateSolverVelocityIterationCountAttr(8)
    api.CreateMaxDepenetrationVelocityAttr(0.2)
    if fixed:
        joint = UsdPhysics.FixedJoint.Define(stage, path + '/diagnostic_mount')
        joint.CreateBody1Rel().SetTargets([root.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pos))
    return root


def shape(stage, path, kind, pos, size, p, color=(.65, .65, .7)):
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    cls = {'sphere': UsdGeom.Sphere, 'cylinder': UsdGeom.Cylinder, 'capsule': UsdGeom.Capsule, 'box': UsdGeom.Cube}[kind]
    geom = cls.Define(stage, path)
    geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if kind == 'sphere':
        geom.CreateRadiusAttr(size)
    elif kind in ('cylinder', 'capsule'):
        geom.CreateRadiusAttr(size[0])
        geom.CreateHeightAttr(size[1])
        geom.CreateAxisAttr('Z')
    else:
        geom.CreateSizeAttr(1.)
        geom.AddScaleOp().Set(Gf.Vec3d(*size))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(p['contact_offset_m'])
    api.CreateRestOffsetAttr(p['rest_offset_m'])
    mat = UsdShade.Material.Define(stage, path + '/material')
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(p['friction'])
    physics.CreateDynamicFrictionAttr(p['friction'])
    physics.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    return geom


def coupon(stage, path, center, p, color, rotation=0., pivot=None):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt, PhysxSchema
    nx = round(p['coupon_width_m'] / p['mesh_pitch_m']) + 1
    ny = round(p['coupon_length_m'] / p['mesh_pitch_m']) + 1
    points = []
    faces = []
    for j in range(ny):
        for i in range(nx):
            points.append([center[0] + (i / (nx - 1) - .5) * p['coupon_width_m'],
                           center[1] + (j / (ny - 1) - .5) * p['coupon_length_m'], center[2]])
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.extend([[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]])
    pts = np.array(points)
    if rotation:
        c, s = math.cos(rotation), math.sin(rotation)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pts = (pts - pivot) @ rot.T + pivot
        points = pts.tolist()
    tri = UsdGeom.Mesh.Define(stage, path)
    tri.CreatePointsAttr(Vt.Vec3fArray(points))
    tri.CreateFaceVertexCountsAttr([3] * len(faces))
    tri.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist())
    tri.CreateSubdivisionSchemeAttr('none')
    tri.CreateDoubleSidedAttr(True)
    tri.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = tri.GetPrim()
    for schema in ('OmniPhysicsDeformableBodyAPI', 'OmniPhysicsSurfaceDeformableSimAPI', 'PhysxSurfaceDeformableBodyAPI'):
        assert prim.ApplyAPI(schema), schema
    set_attr(prim, 'omniphysics:restShapePoints', Vt.Vec3fArray(points))
    set_attr(prim, 'omniphysics:restTriVtxIndices', [Gf.Vec3i(*f) for f in faces])
    set_attr(prim, 'omniphysics:mass', p['coupon_width_m'] * p['coupon_length_m'] * p['areal_density_kg_m2'])
    set_attr(prim, 'physxDeformableBody:selfCollision', True)
    set_attr(prim, 'physxDeformableBody:enableSpeculativeCCD', True)
    set_attr(prim, 'physxDeformableBody:collisionPairUpdateFrequency', 4)
    set_attr(prim, 'physxDeformableBody:solverPositionIterationCount', p['solver_iterations'])
    UsdPhysics.CollisionAPI.Apply(prim)
    cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    cp.CreateContactOffsetAttr(p['contact_offset_m'])
    cp.CreateRestOffsetAttr(p['rest_offset_m'])
    mat = UsdShade.Material.Define(stage, path + '/fabric')
    for api in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI',
                'OmniPhysicsSurfaceDeformableMaterialAPI', 'PhysxSurfaceDeformableMaterialAPI'):
        assert mat.GetPrim().ApplyAPI(api), api
    for name, value in {
        'omniphysics:density': p['areal_density_kg_m2'] / p['thickness_m'],
        'omniphysics:dynamicFriction': p['friction'],
        'omniphysics:youngsModulus': p['youngs_modulus_Pa'],
        'omniphysics:poissonsRatio': p['poisson_ratio'],
        'omniphysics:surfaceThickness': p['thickness_m'],
        'omniphysics:surfaceBendStiffness': p['bend_modulus_Pa'],
        'physxDeformableMaterial:elasticityDamping': .005,
        'physxDeformableMaterial:bendDamping': .005,
    }.items():
        set_attr(mat.GetPrim(), name, value)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    return pts, np.array(faces, dtype=np.int32)


def rivet(stage, path, cloth, hardware, points, center, radius=.0055, rotation=0.):
    """Permanent local rivet attachment; never used for gripper contacts."""
    from pxr import Gf
    local = points - np.array(center)
    if rotation:
        c, s = math.cos(rotation), math.sin(rotation)
        local = local @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    ids = np.flatnonzero(np.linalg.norm(local[:, :2], axis=1) < radius)
    att = stage.DefinePrim(path, 'OmniPhysicsVtxXformAttachment')
    att.GetRelationship('omniphysics:src0').SetTargets([cloth])
    att.GetRelationship('omniphysics:src1').SetTargets([hardware])
    set_attr(att, 'omniphysics:attachmentEnabled', True)
    set_attr(att, 'omniphysics:vtxIndicesSrc0', ids.tolist())
    set_attr(att, 'omniphysics:localPositionsSrc1', [Gf.Vec3f(*local[i]) for i in ids])
    return ids, local[ids]


def build(stage, cfg, offset=0., angle=0., guided=True, cloth=True):
    from pxr import Gf, UsdPhysics, UsdGeom
    p = cfg['snap']
    origin = np.array([0., 0., .1])
    socket = body(stage, '/World/socket', origin, p['socket_mass_kg'], fixed=True)
    # Ring assembled from boxes; the central hole remains empty collision geometry.
    for i in range(24):
        a = i * 2 * math.pi / 24
        s = shape(stage, f'/World/socket/ring{i}', 'box',
                  (p['ring_radius_m'] * math.cos(a), p['ring_radius_m'] * math.sin(a), p['ring_height_m']/2),
                  (p['ring_radial_width_m'], .0014, p['ring_height_m']), p)
        s.AddOrientOp().Set(Gf.Quatf(math.cos(a / 2), 0, 0, math.sin(a / 2)))
    for i in range(4):
        a = i * math.pi / 2
        direction = np.array([math.cos(a), math.sin(a), 0.])
        relative = direction * p['socket_ball_radial_m'] + [0, 0, p['socket_ball_height_m']]
        path = f'/World/ball{i}'
        ball = body(stage, path, origin + relative, p['ball_mass_kg'])
        shape(stage, path + '/contact', 'sphere', (0, 0, 0), p['socket_ball_radius_m'], p)
        joint = UsdPhysics.PrismaticJoint.Define(stage, path + '/spring')
        joint.CreateBody0Rel().SetTargets([socket.GetPath()])
        joint.CreateBody1Rel().SetTargets([ball.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*relative))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        rot = Gf.Quatf(math.cos(a / 2), 0, 0, math.sin(a / 2))
        joint.CreateLocalRot0Attr(rot)
        joint.CreateLocalRot1Attr(rot)
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(0.)
        joint.CreateUpperLimitAttr(p['spring_travel_m'])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(p['spring_stiffness_N_m'])
        drive.CreateDampingAttr(p['spring_damping_Ns_m'])
        drive.CreateTargetPositionAttr(0.)
        drive.CreateMaxForceAttr(100.)
    male_pos = origin + [offset, 0, cfg['fixture']['initial_height_m']]
    male = body(stage, '/World/male', male_pos, p['male_mass_kg'], rotation=angle)
    shape(stage, '/World/male/cap', 'cylinder', (0, 0, 0), (p['cap_diameter_m']/2, p['cap_thickness_m']), p)
    shape(stage, '/World/male/neck', 'cylinder', (0, 0, -.0016), (p['neck_radius_m'], .0032), p)
    shape(stage, '/World/male/head', 'sphere', (0, 0, -p['head_from_cap_m']), p['head_radius_m'], p)
    if guided:
        # Diagnostic load frame. It is not present in the proposed robot task.
        guide = UsdPhysics.PrismaticJoint.Define(stage, '/World/diagnostic_axial_guide')
        guide.CreateBody1Rel().SetTargets([male.GetPath()])
        guide.CreateAxisAttr('Z')
        guide.CreateLocalPos0Attr(Gf.Vec3f(*male_pos))
        guide.CreateLocalRot0Attr(Gf.Quatf(math.cos(angle / 2), 0, math.sin(angle / 2), 0))
        guide.CreateLowerLimitAttr(-.012)
        guide.CreateUpperLimitAttr(.030)
    data = {'male_initial': male_pos.tolist(), 'socket_origin': origin.tolist(), 'cloth': {}}
    if cloth:
        for name, center, color in (('lower', origin + [0, 0, -.0007], (.12, .32, .7)),
                                    ('upper', male_pos + [0, 0, .0007], (.8, .3, .15))):
            rotation = angle if name == 'upper' else 0.
            points, faces = coupon(stage, '/World/' + name, center, cfg['fabric'], color, rotation, male_pos)
            hardware = '/World/socket' if name == 'lower' else '/World/male'
            hardware_center = origin if name == 'lower' else male_pos
            ids, local = rivet(stage, '/World/' + name + '_rivet', '/World/' + name, hardware, points, hardware_center, rotation=rotation)
            data['cloth'][name] = {'initial': points, 'faces': faces, 'rivet_ids': ids, 'local_rivet': local}
    UsdGeom.Xform.Define(stage, '/World/diagnostic_floor')
    shape(stage, '/World/diagnostic_floor/shape', 'box', (0, 0, .03), (.3, .3, .01), p, (.25, .25, .25))
    return data


def build_grasp(stage, cfg):
    """Frictional coupon pinch fixture; no robot or grasp attachments."""
    from pxr import UsdPhysics, Gf, PhysxSchema, UsdGeom
    p = cfg['snap']
    fixture = cfg['pinch_fixture']
    pad_height = fixture['pad_center_height_m']
    center = np.array([0., 0., .1])
    body(stage, '/World/male', center, p['male_mass_kg'], rotation=math.pi/2)
    shape(stage, '/World/male/cap', 'cylinder', (0, 0, 0), (p['cap_diameter_m']/2, p['cap_thickness_m']), p)
    # Coupon is vertical. The top edge lies inside the pad faces, below their
    # top edges, so the frictionless control cannot succeed by draping over them.
    points, faces = coupon(stage, '/World/upper', center + [0, 0, .0007], cfg['fabric'], (.8, .3, .15), math.pi/2, center)
    ids, local = rivet(stage, '/World/upper_rivet', '/World/upper', '/World/male', points, center, rotation=math.pi/2)
    pad_material = dict(p, friction=cfg['fabric']['friction'])
    UsdGeom.Xform.Define(stage, '/World/pinch')
    base = body(stage, '/World/pinch/base', (0, 0, pad_height), .1, fixed=True)
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(base.GetPrim()).CreateEnabledSelfCollisionsAttr(False)
    for i, side in enumerate((-1, 1)):
        carriage = body(stage, f'/World/pinch/carriage{i}', (side*.005, 0, pad_height), .1)
        pad = body(stage, f'/World/pinch/pad{i}', (side*.005, 0, pad_height), .02)
        shape(stage, f'/World/pinch/pad{i}/contact', 'box', (0, 0, 0), fixture['pad_size_m'], pad_material, (.15, .15, .15))
        for name, parent, child, axis, anchor in (
            (f'lift{i}', base, carriage, 'Z', (side*.005, 0, 0)),
            (f'close{i}', carriage, pad, 'X', (0, 0, 0)),
        ):
            joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/pinch/' + name)
            joint.CreateBody0Rel().SetTargets([parent.GetPath()])
            joint.CreateBody1Rel().SetTargets([child.GetPath()])
            joint.CreateAxisAttr(axis)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateLowerLimitAttr(-.02)
            joint.CreateUpperLimitAttr(.04)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(fixture['drive_stiffness_N_m'])
            drive.CreateDampingAttr(fixture['drive_damping_Ns_m'])
            drive.CreateMaxForceAttr(fixture['close_max_force_N'] if axis == 'X' else fixture['lift_max_force_N'])
            drive.CreateTargetPositionAttr(0.)
    # Initial support touches only the cap's lower edge; cloth at x=0.7 mm
    # stays outside the support, whose right edge is x=0.2 mm.
    shape(stage, '/World/cap_support', 'box', (-.0028, 0, .084), (.006, .014, .02), p)
    shape(stage, '/World/floor', 'box', (0, 0, .03), (.3, .3, .01), p)
    return {'male_initial': center.tolist(), 'socket_origin': [0., 0., .1],
            'cloth': {'upper': {'initial': points, 'faces': faces, 'rivet_ids': ids, 'local_rivet': local}}}
