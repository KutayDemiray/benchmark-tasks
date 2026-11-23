"""Movable snap hardware and cloth for ordinary-YAM isolated tests."""

import math
import numpy as np

from geometry import body, coupon, shape, set_attr


def rotation(q):
    """Return the active rotation for an XYZW quaternion."""
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rigid(stage, path, pos, q, mass, cfg):
    """Create a free rigid body; only physical local rivets may attach it."""
    from pxr import Gf, UsdGeom, PhysxSchema
    root = body(stage, path, pos, mass)
    orient = [o for o in root.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeOrient][0]
    orient.Set(Gf.Quatf(float(q[3]), *map(float, q[:3])))
    PhysxSchema.PhysxRigidBodyAPI(root.GetPrim()).CreateMaxAngularVelocityAttr(
        math.degrees(cfg['simulation']['snap_max_angular_speed_rad_s']))
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.)
    return root


def male(stage, path, pos, q, cfg, cap_only=False):
    """Create a male press-stud at a freely moving pose."""
    p = cfg['snap']
    root = rigid(stage, path, pos, q, p['male_mass_kg'], cfg)
    shape(stage, path+'/cap', 'cylinder', (0, 0, 0),
          (p['cap_diameter_m']/2, p['cap_thickness_m']), p)
    if not cap_only:
        shape(stage, path+'/neck', 'cylinder', (0, 0, -.0016), (p['neck_radius_m'], .0032), p)
        shape(stage, path+'/head', 'sphere', (0, 0, -p['head_from_cap_m']), p['head_radius_m'], p)
    return root


def socket(stage, path, pos, q, cfg):
    """Create the same passive p07 socket with its springs in the socket frame."""
    from pxr import Gf, UsdPhysics
    p = cfg['snap']
    root = rigid(stage, path, pos, q, p['socket_mass_kg'], cfg)
    for i in range(24):
        a = i*2*math.pi/24
        item = shape(stage, path+f'/ring{i}', 'box',
                     (p['ring_radius_m']*math.cos(a), p['ring_radius_m']*math.sin(a), p['ring_height_m']/2),
                     (p['ring_radial_width_m'], .0014, p['ring_height_m']), p)
        item.AddOrientOp().Set(Gf.Quatf(math.cos(a/2), 0, 0, math.sin(a/2)))
    balls = []
    for i in range(4):
        a = i*math.pi/2
        relative = np.array([math.cos(a)*p['socket_ball_radial_m'],
                             math.sin(a)*p['socket_ball_radial_m'], p['socket_ball_height_m']])
        ball_path = path+f'_ball{i}'
        ball = rigid(stage, ball_path, np.asarray(pos)+rotation(q)@relative, q, p['ball_mass_kg'], cfg)
        shape(stage, ball_path+'/contact', 'sphere', (0, 0, 0), p['socket_ball_radius_m'], p)
        joint = UsdPhysics.PrismaticJoint.Define(stage, ball_path+'/spring')
        joint.CreateBody0Rel().SetTargets([root.GetPath()])
        joint.CreateBody1Rel().SetTargets([ball.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, relative)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(a/2), 0, 0, math.sin(a/2)))
        joint.CreateLocalRot1Attr(Gf.Quatf(math.cos(a/2), 0, 0, math.sin(a/2)))
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(0.)
        joint.CreateUpperLimitAttr(p['spring_travel_m'])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(p['spring_stiffness_N_m'])
        drive.CreateDampingAttr(p['spring_damping_Ns_m'])
        drive.CreateTargetPositionAttr(0.)
        drive.CreateMaxForceAttr(100.)
        balls.append(ball_path)
    return root, balls


def cloth(stage, path, center, q, hardware_path, hardware_pos, hardware_q, cfg):
    """Create a flexible coupon with a local permanent hardware rivet."""
    from pxr import Gf, UsdGeom, Vt
    p = cfg['fabric']
    points, faces = coupon(stage, path, (0., 0., 0.), p, (.18, .4, .75))
    points = points@rotation(q).T + np.asarray(center)
    prim = stage.GetPrimAtPath(path)
    UsdGeom.Mesh(prim).GetPointsAttr().Set(Vt.Vec3fArray(points.tolist()))
    set_attr(prim, 'omniphysics:restShapePoints', Vt.Vec3fArray(points.tolist()))
    local = (points-np.asarray(hardware_pos))@rotation(hardware_q)
    ids = np.flatnonzero(np.linalg.norm(local[:, :2], axis=1) < .0055)
    assert len(ids) >= 5, 'Hardware needs a resolved local fabric patch'
    att = stage.DefinePrim(path+'_rivet', 'OmniPhysicsVtxXformAttachment')
    att.GetRelationship('omniphysics:src0').SetTargets([path])
    att.GetRelationship('omniphysics:src1').SetTargets([hardware_path])
    set_attr(att, 'omniphysics:attachmentEnabled', True)
    set_attr(att, 'omniphysics:vtxIndicesSrc0', ids.tolist())
    set_attr(att, 'omniphysics:localPositionsSrc1', [Gf.Vec3f(*map(float, v)) for v in local[ids]])
    edges = np.unique(np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1), axis=0)
    return dict(path=path, initial=points, faces=faces, edges=edges,
                rest_lengths=np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1),
                rivet_ids=ids, local_rivet=local[ids], hardware_path=hardware_path)


def grasp_scene(stage, root, cfg):
    """A free coupon over the edge of a passive raised tabletop."""
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, root)
    shape(stage, root+'/floor', 'box', (0., 0., -.025), (4., 4., .05), cfg['snap'], (.25, .25, .25))
    p = cfg['robot_test']
    h = p['support_top_m']
    # The edge is at x=.31; the cloth's free edge can be pinched from +X approach.
    shape(stage, root+'/support', 'box', (.51, .12, h-.02), (.4, .35, .04),
          cfg['snap'], (.40, .32, .24))
    pos = np.array([.325, .12, h+.0005])
    q = [0., 0., 0., 1.]
    male(stage, root+'/male', pos, q, cfg, cap_only=True)
    data = cloth(stage, root+'/cloth', (.3125, .12, h+.0012), q,
                 root+'/male', pos, q, cfg)
    return {'root': root, 'hardware': {'male': root+'/male'}, 'cloth': {'coupon': data},
            'initial_hardware': {'male': [*pos, *q]}, 'support_top_m': h,
            'support_front_x_m': .31, 'diagnostic_world_mounts': False,
            'grasp_attachments': False, 'snap_pair_present': False}


def press_scene(stage, root, cfg):
    """Free cloth-mounted snap halves above a flat physical table, without guides."""
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, root)
    h = cfg['robot_test'].get('press_support_top_m', .90)
    shape(stage, root+'/floor', 'box', (0., 0., -.025), (4., 4., .05), cfg['snap'], (.25, .25, .25))
    lift = cfg['robot_test'].get('case')=='press_lift'
    front_x = .285 if lift else .20
    thickness = cfg['robot_test'].get('support_thickness_m', .04)
    width = cfg['robot_test'].get('support_width_m', .35)
    shape(stage, root+'/support', 'box', (front_x+.20, .12, h-thickness/2),
          (.4, width, thickness), cfg['snap'], (.4, .32, .24))
    if width < .12:
        # A near support keeps this narrow physical beam stiff under pressing.
        shape(stage, root+'/support_post', 'box', (front_x+.10, .12, (h-thickness)/2),
              (.04, .04, h-thickness), cfg['snap'], (.3, .3, .3))
    elif thickness < .04:
        for i, y in enumerate((-.015, .255)):
            shape(stage, root+f'/support_post_{i}', 'box', (front_x+.32, y, (h-thickness)/2),
                  (.04, .04, h-thickness), cfg['snap'], (.3, .3, .3))
    q = [0., 0., 0., 1.]
    sp = np.array([.30, .12, h+.001])
    mp = sp+[0., 0., cfg['fixture']['initial_height_m']]
    _, balls = socket(stage, root+'/socket', sp, q, cfg)
    male(stage, root+'/male', mp, q, cfg)
    lower = cloth(stage, root+'/lower', sp+[.0125 if lift else 0., 0., -.0007], q, root+'/socket', sp, q, cfg)
    import copy
    upper_cfg = copy.deepcopy(cfg)
    upper_width = cfg['robot_test'].get('upper_width_m', cfg['fabric']['coupon_width_m'])
    upper_cfg['fabric']['coupon_width_m'] = upper_width
    upper_center_x = .0075-upper_width/2 if lift else 0.
    upper = cloth(stage, root+'/upper', mp+[upper_center_x, 0., .0007], q,
                  root+'/male', mp, q, upper_cfg)
    hardware = {'male': root+'/male', 'socket': root+'/socket'}
    hardware.update({f'ball{i}': p for i, p in enumerate(balls)})
    return {'root': root, 'hardware': hardware, 'cloth': {'lower': lower, 'upper': upper},
            'initial_hardware': {'male': [*mp, *q], 'socket': [*sp, *q]},
            'support_top_m': h, 'diagnostic_world_mounts': False, 'grasp_attachments': False,
            'snap_pair_present': True, 'initial_alignment_supplied': True,
            'support_front_x_m': front_x, 'support_thickness_m': thickness,
            'support_width_m': width, 'support_center_y_m': .12,
            'grasp_tab_cutoff_x_m': .278, 'upper_coupon_width_m': upper_width}
