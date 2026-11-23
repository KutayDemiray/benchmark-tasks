"""Replay native rigid-frame poses and compare a matched contact negative control."""

from task_paths import OUTPUT_ROOT

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def _directions_unique(vectors):
    """Discard parallel SAT axes; opposite signs describe the same projection."""
    vectors=np.asarray(vectors,dtype=float)
    vectors=vectors[np.linalg.norm(vectors,axis=1)>1e-12]
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    pivot=np.argmax(abs(vectors),axis=1)
    vectors*=np.sign(vectors[np.arange(len(vectors)),pivot])[:,None]
    _,indices=np.unique(np.round(vectors,10),axis=0,return_index=True)
    return vectors[indices]


def features(points):
    """Return unique face normals and true convex-boundary edge directions."""
    hull=ConvexHull(points)
    edges={}
    for index,face in enumerate(hull.simplices):
        for a,b in itertools.combinations(face,2):
            edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(index)
    directions=[]
    for (a,b),faces in edges.items():
        # Coplanar triangle diagonals are not polyhedron edges. Removing them
        # avoids thousands of redundant edge-cross-edge axes for capsules.
        if len(faces)==2 and np.linalg.norm(np.cross(hull.equations[faces[0],:3],hull.equations[faces[1],:3]))<1e-10:
            continue
        directions.append(points[b]-points[a])
    return _directions_unique(hull.equations[:,:3]),_directions_unique(directions)


def penetration(a,b,upper_bound_tolerance=0.):
    """Return convex translation depths, or conservative bounds when requested."""
    from sat_accelerated import depths
    pa,ra,va,na,ea=a
    pb,rb,vb,nb,eb=b
    wa=np.einsum('tij,pj->tpi',ra,va)+pa[:,None]
    wb=np.einsum('tij,pj->tpi',rb,vb)+pb[:,None]
    broad=np.all(wa.max(1)>=wb.min(1),axis=1)&np.all(wb.max(1)>=wa.min(1),axis=1)
    result=np.zeros(len(pa))
    indices=np.flatnonzero(broad)
    if len(indices):
        result[indices]=depths(wa[indices],wb[indices],
            np.einsum('tij,pj->tpi',ra[indices],na),np.einsum('tij,pj->tpi',rb[indices],nb),
            np.einsum('tij,pj->tpi',ra[indices],ea),np.einsum('tij,pj->tpi',rb[indices],eb),
            upper_bound_tolerance)
    return result


def capsule_proxy(capsule):
    """Return a conservative convex capsule bound with <1 mm radial excess.

    An inflated once-subdivided icosahedron contains the true sphere. Its
    Minkowski sum with the capsule segment therefore contains the true capsule.
    Passing a penetration bound on this outer proxy cannot hide a real overlap.
    """
    phi=(1+np.sqrt(5.))/2
    unit=[]
    for a in (-1.,1.):
        for b in (-phi,phi):
            unit.extend([(0.,a,b),(a,b,0.),(b,0.,a)])
    unit=np.asarray(unit)
    unit/=np.linalg.norm(unit,axis=1)[:,None]
    hull=ConvexHull(unit)
    edges=set()
    for tri in hull.simplices:
        edges.update(tuple(sorted(pair)) for pair in itertools.combinations(tri,2))
    mid=np.array([unit[a]+unit[b] for a,b in edges])
    mid/=np.linalg.norm(mid,axis=1)[:,None]
    unit=np.r_[unit,mid]
    apothem=np.min(-ConvexHull(unit).equations[:,3])
    sphere=unit*capsule["radius"]/apothem
    offset=np.array([0.,0.,capsule["height"]/2])
    vertices=np.r_[sphere-offset,sphere+offset]+capsule["center"]
    excess=capsule["radius"]*(1/apothem-1)
    assert excess<.001
    return vertices,float(excess)


def audit(run):
    metadata = json.loads((run/"mechanism.json").read_text())
    request = json.loads((run/"request.json").read_text())
    result = json.loads((run/"result.json").read_text())
    assert result["status"] == "completed"
    data = np.load(run/"states.npz")
    bodies = data["body_poses"]
    names = list(data["body_names"])
    if "pedal_pusher" in metadata:
        for label,size in [("Finger",[.020,.025,.010]),("Base",[.020,.020,.030])]:
            body_name="PedalPusher"+label
            index=list(data["pusher_body_names"]).index(label)
            bodies=np.concatenate((bodies,data["pusher_poses"][:,index:index+1]),axis=1)
            names.append(body_name)
            vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*size
            metadata["colliders"].append(dict(body=body_name,name=label,points=vertices.tolist(),enabled=True))
    shapes = []
    capsule_excess=0.
    floor_penetration = 0.
    for shape in metadata["colliders"]:
        if not shape["enabled"]:
            continue
        pose = bodies[:, names.index(shape["body"])]
        rotations = Rotation.from_quat(pose[:, 3:7]).as_matrix()
        if "capsule" in shape:
            c = shape["capsule"]
            ends = np.array([c["center"], c["center"]], dtype=float)
            ends[:, 2] += np.array([-.5, .5])*c["height"]
            world = np.einsum("tij,pj->tpi", rotations, ends)+pose[:,None,:3]
            floor_penetration = max(floor_penetration, float(c["radius"]-world[:,:,2].min()))
            vertices,excess=capsule_proxy(c)
            capsule_excess=max(capsule_excess,excess)
        else:
            vertices = np.asarray(shape["points"])
        normals, edges = features(vertices)
        shapes.append((shape, (pose[:,:3], rotations, vertices, normals, edges)))
        world = np.einsum("tij,pj->tpi", rotations, vertices)+pose[:,None,:3]
        floor_penetration = max(floor_penetration, float(-world[:,:,2].min()))
    adjacent = [("Bridge","Left"), ("Bridge","Right"),
                ("Bridge","Connector"), ("Connector","Handle"), ("Rocker","Clip")]
    adjacent = {frozenset(pair) for pair in adjacent}
    pairs = []
    for (sa, a), (sb, b) in itertools.combinations(shapes, 2):
        pair = frozenset((sa["body"],sb["body"]))
        if len(pair) == 1 or pair in adjacent:
            continue
        depths = penetration(a,b)
        peak = float(depths.max())
        if peak > 1e-6:
            pairs.append(dict(first=sa["body"]+"/"+sa["name"],
                              second=sb["body"]+"/"+sb["name"], max_depth_m=peak,
                              violating_steps=np.flatnonzero(depths>.001).tolist()))
    maximum = max([p["max_depth_m"] for p in pairs], default=0.)
    joint_names = list(data["joint_names"])
    q = data["joint_positions"]
    hz = request["physics_hz"]
    wing = np.rad2deg(q[:, [joint_names.index("HingeLeft"),joint_names.index("HingeRight")]])
    retention=slice(10*hz,round((12+request.get("retention_extension",0.))*hz))
    flat = np.max(np.abs(wing[retention]))
    gear = np.max(np.abs(wing.sum(1)))
    press = q[:,joint_names.index("LaboratoryPress")]
    lift = float(press[retention].mean() - press[4*hz:6*hz].mean())
    report = dict(scope="Independent rigid mechanism diagnostic; no robot or fabric.",
        diagnostic_only=True, full_task_success=False,
        physics_steps=len(q), floor_penetration_m=max(floor_penetration,0.),
        nonadjacent_convex_penetration_m=maximum, overlapping_pairs=pairs,
        max_gear_error_deg=float(gear), final_two_seconds_wing_angle_deg=float(flat),
        retention_window_s=[10,12+request.get("retention_extension",0.)],capsule_outer_bound_radial_excess_m=capsule_excess,
        physical_unlock_observed=bool(request.get("mechanism_unlock",False) and np.min(abs(wing[-hz:]))>30.),
        final_unlocked_min_wing_angle_deg=float(np.min(abs(wing[-hz:]))) if request.get("mechanism_unlock",False) else None,
        actual_press_lift_m=lift, retained=bool(flat<=3. and lift>.10),
        rigid_checks_passed=bool(maximum<=.001 and floor_penetration<=.001 and gear<=1.),
        limitations=["Capsule/floor checked analytically; capsule/frame uses a conservative convex outer bound.",
                     "Adjacent rigid body overlaps at hinge assemblies are excluded.",
                     "Full mechanism gate also requires contact-negative control and physical unlock test."])
    (run/"mechanism-audit.json").write_text(json.dumps(report,indent=2))
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    args=p.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT_ROOT)
    report=audit(args.run)
    report["overlapping_pairs"]=[{**p,"violating_steps":len(p["violating_steps"])} for p in report["overlapping_pairs"]]
    print(json.dumps(report,indent=2))
