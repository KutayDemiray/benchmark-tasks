"""Native PhysX FEM surface authoring; all calls occur before stepping.

API usage was studied in the existing peg-out-a-tshirt/native_cloth.py and the
installed Isaac Lab deformable tests. No dependency on another task is loaded.
"""

import numpy as np


def set_attr(prim, name, value):
    attr = prim.GetAttribute(name)
    assert attr, f"Missing installed schema attribute: {name}"
    assert attr.Set(value), name


def cloth(stage, points, faces, p, iterations, path="/World/Pocket", color=(0.15, 0.4, 0.75)):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points.tolist()))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = mesh.GetPrim()
    for name in ("OmniPhysicsDeformableBodyAPI", "OmniPhysicsSurfaceDeformableSimAPI", "PhysxSurfaceDeformableBodyAPI"):
        assert prim.ApplyAPI(name), name
    from mesh import material_rest
    rest=material_rest(points,p.get("flat_rest_panels",False),p.get("seam_height_m",0.))
    set_attr(prim, "omniphysics:restShapePoints", Vt.Vec3fArray(rest.tolist()))
    set_attr(prim, "omniphysics:restTriVtxIndices", [Gf.Vec3i(*map(int, t)) for t in faces])
    set_attr(prim, "omniphysics:restBendAnglesDefault", "restShapeDefault")
    if p.get("flat_rest_bending",False):
        from mesh import sewn_bend_pairs
        pairs=sewn_bend_pairs(points,faces)
        set_attr(prim,"omniphysics:restBendAnglesDefault","flatDefault")
        set_attr(prim,"omniphysics:restAdjTriPairs",[Gf.Vec2i(*map(int,pair)) for pair in pairs])
        set_attr(prim,"omniphysics:restBendAngles",[180.]*len(pairs))
    set_attr(prim, "physxDeformableBody:selfCollision", True)
    set_attr(prim, "physxDeformableBody:disableGravity", False)
    set_attr(prim, "physxDeformableBody:collisionPairUpdateFrequency", p.get("collision_pair_updates",1))
    set_attr(prim, "physxDeformableBody:collisionIterationMultiplier", p.get("collision_subiterations",1))
    if p.get("speculative_ccd",False):
        set_attr(prim, "physxDeformableBody:enableSpeculativeCCD", True)
    set_attr(prim, "physxDeformableBody:solverPositionIterationCount", iterations)
    UsdPhysics.CollisionAPI.Apply(prim)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(p["contact_offset_m"])
    collision.CreateRestOffsetAttr(p["rest_offset_m"])
    material = UsdShade.Material.Define(stage, path + "/Fabric")
    for name in ("OmniPhysicsBaseMaterialAPI", "OmniPhysicsDeformableMaterialAPI", "OmniPhysicsSurfaceDeformableMaterialAPI", "PhysxSurfaceDeformableMaterialAPI"):
        assert material.GetPrim().ApplyAPI(name), name
    attrs = {"omniphysics:density": p["density_kg_m3"], "omniphysics:dynamicFriction": p["dynamic_friction"],
             "omniphysics:youngsModulus": p["young_modulus_pa"], "omniphysics:poissonsRatio": p["poisson_ratio"],
             "omniphysics:surfaceThickness": p["shell_thickness_m"], "omniphysics:surfaceBendStiffness": p["bend_modulus_pa"],
             "physxDeformableMaterial:elasticityDamping": p["elasticity_damping"], "physxDeformableMaterial:bendDamping": p["bend_damping"]}
    for name, value in attrs.items():
        set_attr(material.GetPrim(), name, value)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return mesh


def sew_lip(stage, parts, shared, *, filter_mode="element", volume_faces=None, enabled=True, grouping="pairs"):
    """Create only predetermined material-to-material sewing, never capture."""
    from pxr import Sdf, UsdPhysics

    prim = stage.DefinePrim("/World/LipSewing", "OmniPhysicsVtxVtxAttachment")
    prim.GetRelationship("omniphysics:src0").SetTargets([Sdf.Path("/World/Pocket")])
    prim.GetRelationship("omniphysics:src1").SetTargets([Sdf.Path("/World/Lip")])
    for side in range(2):
        lookup = {int(old): new for new, old in enumerate(parts[side][0])}
        offset = len(parts[1][0]) if side == 1 and volume_faces is not None else 0
        set_attr(prim, "omniphysics:vtxIndicesSrc" + str(side), [lookup[int(i)] + offset for i in shared])
    # A fixed seam implements sewing. It cannot capture the holder, move the
    # cloth to a target, or change after startup.
    set_attr(prim, "omniphysics:stiffness", float("inf"))
    set_attr(prim, "omniphysics:attachmentEnabled", enabled)
    if filter_mode == "none":
        return prim
    if filter_mode == "pair":
        # A diagnostic control only: this suppresses real off-seam contact.
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/World/Pocket")).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path("/World/Lip")])
        return prim
    # Adjacent faces across a real sewn seam share material, so contact between
    # those faces is undefined. Mirror the one-ring exclusions of a single
    # mesh. Every other lip/fabric contact, and all frame contacts, stay active.
    filter_prim = stage.DefinePrim("/World/SewnNeighbors", "OmniPhysicsElementCollisionFilter")
    filter_prim.GetRelationship("omniphysics:src0").SetTargets([Sdf.Path("/World/Pocket")])
    filter_prim.GetRelationship("omniphysics:src1").SetTargets([Sdf.Path("/World/Lip")])
    global_faces = [ids[faces] for ids, _, faces in parts]
    if volume_faces is not None:
        global_faces[1] = parts[1][0][volume_faces % len(parts[1][0])]
    if grouping == "vertices":
        # A collision feature shared with an unfiltered face remains active.
        # Group each complete seam vertex star, rather than single face pairs.
        counts = [[], []]
        indices = [[], []]
        for vertex in shared:
            for side in range(2):
                incident = np.flatnonzero(np.any(global_faces[side] == vertex, axis=1))
                assert len(incident)
                counts[side].append(len(incident))
                indices[side].extend(incident.tolist())
        for side in range(2):
            set_attr(filter_prim, "omniphysics:groupElemCounts" + str(side), counts[side])
            set_attr(filter_prim, "omniphysics:groupElemIndices" + str(side), indices[side])
        return prim
    pairs = set()
    for vertex in shared:
        first = np.flatnonzero(np.any(global_faces[0] == vertex, axis=1))
        second = np.flatnonzero(np.any(global_faces[1] == vertex, axis=1))
        for a in first:
            for b in second:
                pairs.add((int(a), int(b)))
    pairs = sorted(pairs)
    for side in range(2):
        set_attr(filter_prim, "omniphysics:groupElemCounts" + str(side), [1] * len(pairs))
        set_attr(filter_prim, "omniphysics:groupElemIndices" + str(side), [pair[side] for pair in pairs])
    return prim


def box(stage, path, size, pos, *, kinematic=False, dynamic=False, mass=0.15, color=(0.4, 0.4, 0.4)):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    root = UsdGeom.Xform.Define(stage, path)
    translate = root.AddTranslateOp()
    translate.Set(Gf.Vec3d(*pos))
    if kinematic or dynamic:
        rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rb.CreateKinematicEnabledAttr(kinematic)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    cube = UsdGeom.Cube.Define(stage, path + "/Shape")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    collision.CreateContactOffsetAttr(0.0005)
    collision.CreateRestOffsetAttr(0.0)
    material = UsdShade.Material.Define(stage, path + "/Material")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(0.55)
    api.CreateDynamicFrictionAttr(0.55)
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return translate


def linear_rig(stage, size, profile="square", start_x=.26):
    """Visible one-axis bench stage with a force-limited native actuator.

    This constrains the test tool only. It has no connection to the fabric and
    must never be used in the robot task as a guide or an automatic inserter.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, Vt

    root = UsdGeom.Xform.Define(stage, "/World/Rig")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    box(stage, "/World/Rig/Base", (0.04, 0.04, 0.04), (start_x+.14, 0, 0.05), dynamic=True, color=(0.7, 0.25, 0.2))
    blade_z = .004 if profile == "beveled" else .006
    box(stage, "/World/Rig/Blade", size, (start_x, 0, blade_z), dynamic=True, color=(0.8, 0.65, 0.08))
    if profile == "beveled":
        from blade_geometry import shape
        vertices, faces = shape(size)
        stage.RemovePrim("/World/Rig/Blade/Shape")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Rig/Blade/Shape")
        mesh.CreatePointsAttr(Vt.Vec3fArray(vertices.tolist()))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.8,.65,.08)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        api.CreateContactOffsetAttr(.0005)
        api.CreateRestOffsetAttr(0.)
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath("/World/Rig/Blade/Material")),
            UsdShade.Tokens.weakerThanDescendants, "physics")
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Rig/Anchor")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/World/Rig/Base")])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(start_x+.14, 0, 0.05))
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/Rig/Slide")
    slide.CreateBody0Rel().SetTargets([Sdf.Path("/World/Rig/Base")])
    slide.CreateBody1Rel().SetTargets([Sdf.Path("/World/Rig/Blade")])
    slide.CreateLocalPos0Attr(Gf.Vec3f(-0.14, 0, blade_z-.05))
    slide.CreateAxisAttr("X")
    slide.CreateLowerLimitAttr(-0.205)
    slide.CreateUpperLimitAttr(0.001)
    drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(2000)
    drive.CreateDampingAttr(30)
    drive.CreateMaxForceAttr(10)
    drive.CreateTargetPositionAttr(0)
    return root


def rubber_volume(stage, points, tets, p, iterations, path="/World/Lip"):
    """Author finite-thickness rubber using the installed volume FEM API."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from omni.physx.scripts import deformableUtils

    mesh = UsdGeom.TetMesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points.tolist()))
    mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray(tets.tolist()))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.2)])
    assert deformableUtils.set_physics_volume_deformable_body(stage, Sdf.Path(path))
    prim = mesh.GetPrim()
    assert prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    set_attr(prim, "physxDeformableBody:selfCollision", True)
    set_attr(prim, "physxDeformableBody:disableGravity", False)
    set_attr(prim, "physxDeformableBody:collisionPairUpdateFrequency", p.get("collision_pair_updates",1))
    set_attr(prim, "physxDeformableBody:collisionIterationMultiplier", p.get("collision_subiterations",1))
    if p.get("speculative_ccd",False):
        set_attr(prim, "physxDeformableBody:enableSpeculativeCCD", True)
    set_attr(prim, "physxDeformableBody:solverPositionIterationCount", iterations)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(p["contact_offset_m"])
    # Geometric volume supplies thickness; do not add a second shell half-thickness.
    collision.CreateRestOffsetAttr(0.0)
    material = UsdShade.Material.Define(stage, path + "/Rubber")
    for name in ("OmniPhysicsBaseMaterialAPI", "OmniPhysicsDeformableMaterialAPI", "PhysxDeformableMaterialAPI"):
        assert material.GetPrim().ApplyAPI(name)
    for name, value in {
        "omniphysics:density": p["density_kg_m3"],
        "omniphysics:dynamicFriction": p["dynamic_friction"],
        "omniphysics:youngsModulus": p["young_modulus_pa"],
        "omniphysics:poissonsRatio": p["poisson_ratio"],
        "physxDeformableMaterial:elasticityDamping": p["elasticity_damping"],
    }.items():
        set_attr(material.GetPrim(), name, value)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    boundary = np.asarray(mesh.GetSurfaceFaceVertexIndicesAttr().Get(), dtype=np.int32)
    return mesh, boundary
