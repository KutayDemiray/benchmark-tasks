"""Read-only stock gripper collision geometry and native-pose contact audit."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def collect(stage, root, body_names, selected=("gripper","tip_left","tip_right")):
    """Read each authored convex collider in its owning rigid body's frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache=UsdGeom.XformCache()
    records=[]
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not str(prim.GetPath()).startswith(root+"/") or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body=body.GetParent()
        if not body or body.GetName() not in selected:
            continue
        assert prim.IsA(UsdGeom.Mesh), "Unimplemented stock gripper collider type"
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
        local=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
        hull=ConvexHull(local)
        approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        assert approximation=="convexHull", approximation
        records.append(dict(body=body.GetName(),name=str(prim.GetPath())[len(root):],
                            points=local[hull.vertices].tolist(),planes=hull.equations.tolist()))
    assert set(r["body"] for r in records)==set(selected)
    return dict(body_names=list(selected),indices=[body_names.index(n) for n in selected],
                colliders=records,quaternion_order="xyzw")


def collect_cooked(stage,root,authored):
    """Read PhysX's actual convex representation without changing the scene.

    The installed ConvexMeshDataDemo documents this synchronous cooking query.
    Points returned by it are in the source mesh frame, as in that example.
    """
    import hashlib
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools,PhysxSchema,UsdGeom,UsdUtils,UsdPhysics
    cooking=get_physx_cooking_interface()
    stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cache=UsdGeom.XformCache()
    records=[]
    stage_before=stage.GetRootLayer().ExportToString()
    for original in authored['colliders']:
        path=root+original['name']
        prim=stage.GetPrimAtPath(path)
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        assert body and body.GetName()==original['body']
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
        source_points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        current=(np.c_[source_points,np.ones(len(source_points))]@matrix)[:,:3]
        current_hull=ConvexHull(current)
        original_points=np.asarray(original['points'])
        from scipy.spatial import cKDTree
        current_vertices=current[current_hull.vertices]
        input_error=max(cKDTree(current_vertices).query(original_points)[0].max(),
                        cKDTree(original_points).query(current_vertices)[0].max())
        assert input_error<2e-7, f'Archived authored collider does not match USD input: {path}'
        response=[]
        def received(result,convexes):response.append((result,convexes))
        cooking.request_convex_collision_representation(stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            run_asynchronously=False,on_result=received)
        assert len(response)==1 and response[0][0]==PhysxCollisionRepresentationResult.RESULT_VALID,path
        convexes=response[0][1]
        assert len(convexes)==1, 'Stock convexHull mesh must produce one convex collider'
        source=np.array([[v.x,v.y,v.z] for v in convexes[0].vertices])
        local=(np.c_[source,np.ones(len(source))]@matrix)[:,:3]
        hull=ConvexHull(local)
        outside=(original_points@hull.equations[:,:3].T+hull.equations[:,3]).max()
        records.append(dict(body=original['body'],name=original['name'],
            points=local[hull.vertices].tolist(),planes=hull.equations.tolist(),
            native_vertex_count=len(source),authored_vertex_count=len(original_points),
            archived_geometry_input_error_m=float(input_error),
            authored_points_max_outside_cooked_m=float(outside),
            hull_vertex_limit=PhysxSchema.PhysxConvexHullCollisionAPI(prim).GetHullVertexLimitAttr().Get(),
            authored_points_sha256=hashlib.sha256(original_points.tobytes()).hexdigest()))
    assert stage.GetRootLayer().ExportToString()==stage_before, 'Cooking query modified the source stage'
    return dict(body_names=authored['body_names'],indices=authored['indices'],colliders=records,
        quaternion_order='xyzw',geometry_source='PhysX request_convex_collision_representation',
        scene_modified=False)


def triangles_in_convex(points, faces, equations, tolerance=.001):
    """Count triangles entering the convex interior by the given tolerance."""
    triangles=points[faces]
    normals=equations[:,:3]
    offsets=-equations[:,3]-tolerance
    signed=triangles@normals.T-offsets
    candidates=triangles[~np.any(np.all(signed>0,axis=1),axis=1)]
    hits=0
    for triangle in candidates:
        polygon=list(triangle)
        for normal,offset in zip(normals,offsets):
            if not polygon:
                break
            clipped=[]
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                da,db=a@normal-offset,b@normal-offset
                if da<=0:
                    clipped.append(a)
                if da*db<0:
                    clipped.append(a+(b-a)*da/(da-db))
            polygon=clipped
        hits+=bool(polygon)
    return hits


def penetrating_gripper_shapes(points, faces, poses, geometry):
    """Return stock convex collider names penetrated by native cloth triangles."""
    if "_compiled" not in geometry:
        compiled=[]
        for name in geometry["body_names"]:
            shapes=[]
            for shape in geometry["colliders"]:
                if shape["body"]==name:
                    vertices=np.asarray(shape["points"])
                    shapes.append((shape["name"],vertices.min(0),vertices.max(0),np.asarray(shape["planes"])))
            compiled.append(shapes)
        geometry["_compiled"]=compiled
    hits=[]
    world_low,world_high=points.min(0),points.max(0)
    world_center=(world_low+world_high)/2
    world_half=(world_high-world_low)/2
    for i,shapes in enumerate(geometry["_compiled"]):
        pose=poses[i]
        rotation=Rotation.from_quat(pose[3:7]).as_matrix()
        # Transform an enclosing world box first. It can retain extra shapes,
        # but cannot discard any shape reached by a material triangle.
        center=(world_center-pose[:3])@rotation
        half=world_half@abs(rotation)
        nearby=[s for s in shapes if np.all(center+half>=s[1]) and np.all(center-half<=s[2])]
        if not nearby:
            continue
        local=(points-pose[:3])@rotation
        cloth_low,cloth_high=local.min(0),local.max(0)
        nearby=[s for s in nearby if np.all(cloth_high>=s[1]) and np.all(cloth_low<=s[2])]
        if not nearby:
            continue
        triangles=local[faces]
        low,high=triangles.min(1),triangles.max(1)
        for name,minimum,maximum,equations in nearby:
            selected=np.all(high>=minimum,axis=1)&np.all(low<=maximum,axis=1)
            if selected.any() and triangles_in_convex(local,faces[selected],equations):
                hits.append(name)
    return hits
