"""Fail-closed checks on the actual native environment after reset."""
import hashlib
import json
from pathlib import Path
import numpy as np
from yam_arena import YAM_URDF
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, SHOULDER, WRIST, GRIPPER
from yam_arena.urdf import parse_joints
from adapter import array


def check(env, destination=None):
    """Preserve partial native evidence even when an assertion rejects reset."""
    result = {"passed": False}
    try:
        _check(env, result)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if destination is not None:
            Path(destination).write_text(json.dumps(result, indent=2)+"\n")
    return result


def _check(env, result):
    base = env.unwrapped
    result.update(physics_dt=base.physics_dt, control_dt=base.step_dt,
                  stock_urdf_sha256=hashlib.sha256(YAM_URDF.read_bytes()).hexdigest(), robots={})
    assert abs(base.physics_dt-1/240) < 1e-9 and abs(base.step_dt-.05) < 1e-9
    joints = parse_joints(YAM_URDF)
    for side in ("left_robot", "right_robot"):
        art = base.scene[side]
        records = {}
        for i, name in enumerate(art.joint_names):
            if name not in ARM_JOINTS+FINGER_JOINTS:
                continue
            expected = SHOULDER if name in ARM_JOINTS[:3] else WRIST if name in ARM_JOINTS[3:] else GRIPPER
            values = {}
            for key, prop, wanted in (("stiffness", "joint_stiffness", expected["stiffness"]),
                                      ("damping", "joint_damping", expected["damping"]),
                                      ("effort_limit", "joint_effort_limits", expected["effort"])):
                actual = float(array(getattr(art.data, prop))[0, i])
                assert np.isclose(actual, wanted), (side, name, key, actual, wanted)
                values[key] = actual
            limits = array(art.data.joint_pos_limits)[0, i]
            np.testing.assert_allclose(limits, [joints[name].lower, joints[name].upper], atol=1e-5)
            values["limits_rad_or_m"] = limits.tolist()
            records[name] = values
        result["robots"][side] = records
    tool = base.scene["multitool"]
    assert set(tool.body_names) == {"jaw_a", "jaw_b", "handle_a", "handle_b"}
    assert set(tool.joint_names) == {"jaw", "fold_a", "fold_b"}
    np.testing.assert_allclose(array(tool.data.joint_stiffness), 0.)
    np.testing.assert_allclose(array(tool.data.joint_damping), 0.)
    assert tool.passive_update_count > 0
    from mechanism import load_spec
    from geometry import tool_links, mass_properties
    spec = load_spec()
    masses = array(tool.data.body_mass)[0]
    inertias = array(tool.data.body_inertia)[0].reshape(-1, 3, 3)
    pieces = tool_links(spec)
    for i, name in enumerate(tool.body_names):
        expected_mass = spec["body_masses_kg"][name]
        np.testing.assert_allclose(masses[i], expected_mass, rtol=1e-5)
        _, expected_inertia = mass_properties(pieces[name], expected_mass)
        np.testing.assert_allclose(np.linalg.eigvalsh(inertias[i]),
                                   np.linalg.eigvalsh(expected_inertia), rtol=2e-4, atol=1e-10)
    result["tool"] = {"bodies": list(tool.body_names), "joints": list(tool.joint_names),
                      "actual_masses_kg": dict(zip(tool.body_names, masses.tolist())),
                      "actual_inertia_eigenvalues_kg_m2": np.linalg.eigvalsh(inertias).tolist(),
                      "zero_pd_gains": True, "passive_updates": tool.passive_update_count,
                      "native_static_friction": array(tool.data.joint_friction_coeff)[0].tolist(),
                      "native_viscous_friction": array(tool.data.joint_viscous_friction_coeff)[0].tolist()}
    from adapter import PAYLOAD
    if PAYLOAD is not None:
        payload = base.scene[PAYLOAD.name]
        mass = float(array(payload.data.body_mass).reshape(-1)[0])
        np.testing.assert_allclose(mass, PAYLOAD.mass, atol=1e-7)
        np.testing.assert_allclose(base.cfg.sim.gravity, [0., 0., -9.81])
        result["payload"] = {"name": PAYLOAD.name, "actual_mass_kg": mass,
                             "gravity": list(base.cfg.sim.gravity)}
        view = base.sim.physics_sim_view.create_rigid_body_view("/World/envs/env_0/"+PAYLOAD.name)
        result["payload"]["native_collision_shape_count"] = int(view.max_shapes)
        assert view.count == 1
        assert view.max_shapes == sum(piece.collision for piece in PAYLOAD.pieces)
    from adapter import FIXTURES
    if FIXTURES:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        stage = get_current_stage()
        cache = UsdGeom.XformCache()
        result["fixtures"] = {}
        for fixture in FIXTURES:
            path = "/World/envs/env_0/"+fixture.name
            prim = stage.GetPrimAtPath(path)
            meshes = []
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(child)
                    points = np.asarray(mesh.GetPointsAttr().Get(), float)
                    approximation = UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()
                elif child.IsA(UsdGeom.Cube):
                    half = float(UsdGeom.Cube(child).GetSizeAttr().Get())/2
                    points = np.array([[x, y, z] for x in (-half, half)
                                      for y in (-half, half) for z in (-half, half)])
                    approximation = "native_box"
                else:
                    continue
                matrix = np.asarray(cache.GetLocalToWorldTransform(child), float).T
                world = points@matrix[:3, :3].T+matrix[:3, 3]
                collision = UsdPhysics.CollisionAPI(child)
                meshes.append({"path": str(child.GetPath()), "usd_type": child.GetTypeName(),
                    "low": world.min(0).tolist(), "high": world.max(0).tolist(),
                    "collision_enabled": collision.GetCollisionEnabledAttr().Get(),
                    "approximation": approximation})
            result["fixtures"][fixture.name] = {"world_matrix": np.asarray(cache.GetLocalToWorldTransform(prim)).tolist(),
                "kinematic": UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(),
                "actual_root_pose": array(base.scene[fixture.name].data.root_link_pose_w)[0].tolist(),
                "meshes": meshes}
            entry = result["fixtures"][fixture.name]
            view = base.sim.physics_sim_view.create_rigid_body_view(path)
            entry["native_body_count"] = int(view.count)
            entry["native_collision_shape_count"] = int(view.max_shapes)
            assert view.count == 1, (fixture.name, "missing or ambiguous rigid root")
            assert view.max_shapes == sum(piece.collision for piece in fixture.pieces), (
                fixture.name, "native collision pieces differ from declared geometry")
            assert entry["kinematic"] is True, (fixture.name, "fixture must remain fixed")
            np.testing.assert_allclose(entry["actual_root_pose"][:3], fixture.pos, atol=2e-6, rtol=0)
            assert all(m["collision_enabled"] for m in meshes), (fixture.name, "disabled collider")
            assert len(meshes) == len(fixture.pieces)
            if fixture.box_primitives:
                assert all(m["usd_type"] == "Cube" for m in meshes), (fixture.name, "expected native boxes")
            # Check placement of the actual USD collision meshes, not merely
            # the fixture's root/configuration. A floor in the wrong frame
            # can look plausible while allowing the nut to fall through.
            c, s = np.cos(fixture.yaw), np.sin(fixture.yaw)
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            by_name = {Path(m["path"]).name: m for m in meshes}
            for piece in fixture.pieces:
                expected = piece.vertices@rotation.T+np.asarray(fixture.pos)
                actual = by_name[piece.name]
                np.testing.assert_allclose(actual["low"], expected.min(0), atol=2e-6, rtol=0)
                np.testing.assert_allclose(actual["high"], expected.max(0), atol=2e-6, rtol=0)
    result["passed"] = True
    return result
