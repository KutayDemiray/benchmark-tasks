"""Task-local four-link USD asset, preserving stock YAM and Arena sources."""
from dataclasses import dataclass, field
import math
import numpy as np

from yam_arena.task_file import Box
from mechanism import BODIES, JOINTS, link_transforms, reset_joints
from geometry import tool_links, mass_properties


@dataclass(frozen=True)
class FoldTool(Box):
    spec: dict = field(default_factory=dict, compare=False)


def make_object(spec_object):
    """Build an Arena Object using only installed native APIs."""
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    from passive_actuator import PassiveToolArticulation
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    @configclass
    class PassiveToolCfg(ArticulationCfg):
        class_type: type = PassiveToolArticulation
        mechanics: dict = spec_object.spec["joints"]

    @configclass
    class Spawner(SpawnerCfg):
        func: object = spawn
        mechanism: dict = spec_object.spec

    class ToolObject(Object):
        def _build_reset_event(self):
            return EventTermCfg(func=reset_tool, mode="reset", params={"asset_cfg": SceneEntityCfg(self.name)})

        def _generate_articulation_cfg(self):
            # Resolve Coulomb/viscous loss inside the native implicit solver. An
            # explicit sign-friction torque can inject numerical energy into this
            # small, light linkage at the fixed stock 240 Hz timestep.
            js = spec_object.spec["joints"]
            act = ImplicitActuatorCfg(joint_names_expr=list(JOINTS), stiffness=0., damping=0.,
                                     armature=0., friction={k: js[k]["coulomb_nm"] for k in JOINTS},
                                     dynamic_friction={k: js[k]["coulomb_nm"] for k in JOINTS},
                                     viscous_friction={k: js[k]["viscous_nms"] for k in JOINTS})
            cfg = PassiveToolCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators={"passive": act})
            cfg = self._add_initial_pose_to_cfg(cfg)
            cfg.init_state.joint_pos = reset_joints(spec_object.spec)
            cfg.init_state.joint_vel = {".*": 0.}
            return cfg

    return ToolObject(name=spec_object.name, object_type=ObjectType.ARTICULATION,
                      spawner_cfg=Spawner(), initial_pose=Pose(position_xyz=spec_object.pos,
                      rotation_xyzw=(0., 0., 0., 1.)))


def reset_tool(env, env_ids, asset_cfg):
    """The only explicit tool-state writes, invoked exclusively as a reset event."""
    import torch
    art = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=art.device)
    pose = art.data.default_root_pose.torch[env_ids].clone().to(torch.float32)
    pose[:, :3] += env.scene.env_origins[env_ids]
    q = art.data.default_joint_pos.torch[env_ids].clone()
    art.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    art.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids), 6), device=art.device), env_ids=env_ids)
    art.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=env_ids)


def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
    from yam_arena import ik
    from yam_arena.station import linear_rgb

    spec, stage = cfg.mechanism, get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    x, y, z, w = orientation or (0, 0, 0, 1)
    root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    mat = sim.RigidBodyMaterialCfg(static_friction=spec["contact"]["static_friction"],
                                 dynamic_friction=spec["contact"]["dynamic_friction"], restitution=0.)
    mat.func(prim_path+"/physics", mat)
    links = tool_links(spec)
    poses = link_transforms(spec, reset_joints(spec))
    for name in BODIES:
        path = prim_path+"/"+name
        tr = poses[name]
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*tr[:3, 3]))
        x, y, z, w = ik.matrix_to_quat_xyzw(tr[:3, :3])
        body.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
        rb.CreateLinearDampingAttr(0.)
        rb.CreateAngularDampingAttr(0.)
        rb.CreateMaxDepenetrationVelocityAttr(1.)
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
        mass = spec["body_masses_kg"][name]
        com, inertia = mass_properties(links[name], mass)
        ev, er = np.linalg.eigh(inertia)
        if np.linalg.det(er) < 0:
            er[:, 0] *= -1
        qx, qy, qz, qw = ik.matrix_to_quat_xyzw(er)
        mi = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mi.CreateMassAttr(mass)
        mi.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mi.CreateDiagonalInertiaAttr(Gf.Vec3f(*ev))
        mi.CreatePrincipalAxesAttr(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
        for piece in links[name]:
            mesh = UsdGeom.Mesh.Define(stage, path+"/"+piece.name)
            mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in piece.vertices])
            triangles = piece.triangles
            mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
            mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*linear_rgb(piece.color))])
            if piece.collision:
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                collision.CreateContactOffsetAttr(spec["contact"]["contact_offset_m"])
                collision.CreateRestOffsetAttr(0.)
                bind_physics_material(str(mesh.GetPath()), prim_path+"/physics", stage=stage)
    g = spec["geometry"]
    definitions = [("jaw", "jaw_a", "jaw_b", (0, 0, 0)),
                   ("fold_a", "jaw_a", "handle_a", (g["fold_pivot_x"], -g["fold_pivot_half_span"], 0)),
                   ("fold_b", "jaw_b", "handle_b", (g["fold_pivot_x"], g["fold_pivot_half_span"], 0))]
    for name, parent, child, anchor in definitions:
        j = UsdPhysics.RevoluteJoint.Define(stage, prim_path+"/joints/"+name)
        j.CreateAxisAttr("Z")
        j.CreateBody0Rel().SetTargets([Sdf.Path(prim_path+"/"+parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(prim_path+"/"+child)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1))
        j.CreateLocalRot1Attr(Gf.Quatf(1))
        j.CreateCollisionEnabledAttr(True)
        lo, hi = spec["joints"][name]["limits_rad"]
        j.CreateLowerLimitAttr(math.degrees(lo))
        j.CreateUpperLimitAttr(math.degrees(hi))
        js = PhysxSchema.JointStateAPI.Apply(j.GetPrim(), "angular")
        js.CreatePositionAttr(math.degrees(spec["joints"][name]["reset_rad"]))
        js.CreateVelocityAttr(0.)
    root_body = stage.GetPrimAtPath(prim_path+"/jaw_a")
    UsdPhysics.ArticulationRootAPI.Apply(root_body)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root_body)
    art.CreateEnabledSelfCollisionsAttr(True)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()
