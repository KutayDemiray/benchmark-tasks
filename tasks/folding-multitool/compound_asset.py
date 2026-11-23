"""Task-local compound rigid assets using the native object/reset interfaces."""
from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np
from yam_arena.task_file import Box
from geometry import Piece, mass_properties


@dataclass(frozen=True)
class Compound(Box):
    pieces: tuple = field(default_factory=tuple, compare=False)
    friction: tuple = (.45, .35)
    box_primitives: bool = False


def make_object(spec):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class Spawn(SpawnerCfg):
        func: object = spawn
        # Native configclass recursively rewrites dataclass fields. Box is
        # frozen, so carry plain data through that boundary, not the Box itself.
        compound: dict = dict(free=spec.free, mass=spec.mass, friction=spec.friction,
            box_primitives=spec.box_primitives,
            pieces=[dict(name=p.name, vertices=p.vertices.tolist(), color=p.color,
                         collision=p.collision) for p in spec.pieces])

    # Use the native YAM library-fixture representation for compound fixed
    # objects: one kinematic rigid root groups all convex child colliders.
    # Their declared pose remains fixed, just like a bolted physical fixture.
    return Object(name=spec.name, object_type=ObjectType.RIGID,
                  spawner_cfg=Spawn(), initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                  rotation_xyzw=(0., 0., float(np.sin(spec.yaw/2)), float(np.cos(spec.yaw/2)))))


def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from isaaclab.sim.utils import bind_physics_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    from yam_arena import ik
    from yam_arena.station import linear_rgb

    spec, stage = SimpleNamespace(**cfg.compound), get_current_stage()
    spec.pieces = [Piece(p["name"], np.asarray(p["vertices"]), tuple(p["color"]),
                        p["collision"]) for p in spec.pieces]
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    x, y, z, w = orientation or (0, 0, 0, 1)
    root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    material = sim.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1], restitution=0.)
    material.func(prim_path+"/physics", material)
    body_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body_api.CreateKinematicEnabledAttr(not spec.free)
    if spec.free:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        rb.CreateLinearDampingAttr(0.)
        rb.CreateAngularDampingAttr(0.)
        rb.CreateMaxDepenetrationVelocityAttr(1.)
        PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.)
        com, inertia = mass_properties(spec.pieces, spec.mass)
        ev, er = np.linalg.eigh(inertia)
        if np.linalg.det(er) < 0:
            er[:, 0] *= -1
        qx, qy, qz, qw = ik.matrix_to_quat_xyzw(er)
        mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass.CreateMassAttr(float(spec.mass))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*ev))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    for piece in spec.pieces:
        path = prim_path+"/"+piece.name
        if spec.box_primitives:
            low, high = piece.vertices.min(0), piece.vertices.max(0)
            corners = np.array([[x, y, z] for x in (low[0], high[0])
                                for y in (low[1], high[1]) for z in (low[2], high[2])])
            assert piece.vertices.shape == (8, 3) and set(map(tuple, piece.vertices)) == set(map(tuple, corners))
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.)
            shape.AddTranslateOp().Set(Gf.Vec3d(*((low+high)/2)))
            shape.AddScaleOp().Set(Gf.Vec3f(*(high-low)))
        else:
            shape = UsdGeom.Mesh.Define(stage, path)
            shape.CreatePointsAttr([Gf.Vec3f(*v) for v in piece.vertices])
            triangles = piece.triangles
            shape.CreateFaceVertexCountsAttr([3]*len(triangles))
            shape.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
            shape.CreateSubdivisionSchemeAttr("none")
            shape.CreateDoubleSidedAttr(True)
        shape.CreateDisplayColorAttr([Gf.Vec3f(*linear_rgb(piece.color))])
        if piece.collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            if not spec.box_primitives:
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
            coll = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
            coll.CreateContactOffsetAttr(.0003)
            coll.CreateRestOffsetAttr(0.)
            bind_physics_material(str(shape.GetPath()), prim_path+"/physics", stage=stage)
    return root.GetPrim()
