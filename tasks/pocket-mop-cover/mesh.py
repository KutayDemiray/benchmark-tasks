"""A flexible pocket sewn along three sides; geometry and read-only measurements."""

from collections import Counter

import numpy as np


def pocket_mesh(p, *, spacing=None, mouth_height=None):
    """Return a two-dimensional sewn surface, its faces and material markers.

    Coordinates are metres. X runs from the closed end to the free cloth tail;
    Z is up. Shared vertices implement the actual three sewn boundaries.
    """
    h = spacing or p["mesh_spacing_m"]
    length, width, depth = (p[k] for k in ("coupon_length_m", "coupon_width_m", "pocket_depth_m"))
    height = p["initial_mouth_height_m"] if mouth_height is None else mouth_height
    nx, ny, npocket = (round(v / h) for v in (length, width, depth))
    assert all(abs(n * h - v) < 1e-8 for n, v in zip((nx, ny, npocket), (length, width, depth)))
    seam_height=p.get("seam_height_m",0.)
    points, faces, base, top = [], [], {}, {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            base[i, j] = len(points)
            points.append((i * h, j * h - width / 2, 0.002))
    for i in range(npocket + 1):
        for j in range(ny + 1):
            if not seam_height and (i == 0 or j in (0, ny)):
                top[i, j] = base[i, j]
            else:
                top[i, j] = len(points)
                z = 0.002 + seam_height + height * np.sin(np.pi * j / ny) * np.sin(np.pi * i / (2 * npocket))
                points.append((i * h, j * h - width / 2, z))
    for grid, rows, reverse in ((base, nx, True), (top, npocket, False)):
        for i in range(rows):
            for j in range(ny):
                a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
                pair = [(a, b, c), (a, c, d)]
                if not reverse and i == 0 and j == ny - 1:
                    # The opposite diagonal avoids a face made entirely of
                    # seam nodes coinciding with the base at this corner.
                    pair = [(a, b, d), (b, c, d)]
                for tri in pair:
                    faces.append(tuple(reversed(tri)) if reverse else tri)
    if seam_height:
        # Finite folded seam: sealed side strips connect the two layer midplanes.
        # They are ordinary material faces, with no solver attachment or capture.
        def wall(a,b,c,d):
            faces.extend([(a,b,c),(a,c,d)])
        for j in range(ny):
            wall(base[0,j],top[0,j],top[0,j+1],base[0,j+1])
        for i in range(npocket):
            wall(base[i,0],base[i+1,0],top[i+1,0],top[i,0])
            wall(base[i,ny],top[i,ny],top[i+1,ny],base[i+1,ny])
    # A sewn side ending at the open mouth is a boundary junction of one sheet,
    # not a third sheet attached inside a cloth face.
    markers = {
        "mouth_upper": [top[npocket, j] for j in range(1, ny)],
        "mouth_lower": [base[npocket, j] for j in range(1, ny)],
        "upper": sorted(set(top.values())),
        "base": sorted(set(base.values())),
        "upper_grid": [[top[i, j] for j in range(ny + 1)] for i in range(npocket + 1)],
        "base_grid": [[base[i, j] for j in range(ny + 1)] for i in range(nx + 1)],
        "dimensions_m": [length, width, depth],
        "seam_height_m": seam_height,
    }
    return np.asarray(points), np.asarray(faces, dtype=np.int32), markers


def audit(points, faces):
    """Reject degenerate, disconnected or nonmanifold sewn meshes."""
    assert np.isfinite(points).all()
    assert faces.min() >= 0 and faces.max() < len(points)
    areas = np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]), axis=1) / 2
    edges = Counter()
    adjacency = {i: set() for i in range(len(points))}
    for tri in faces:
        for a, b in zip(tri, np.roll(tri, -1)):
            edges[tuple(sorted((int(a), int(b))))] += 1
            adjacency[a].add(b)
            adjacency[b].add(a)
    assert areas.min() > 1e-12, "Degenerate faces"
    assert max(edges.values()) <= 2, "Nonmanifold seam"
    seen, todo = {0}, [0]
    while todo:
        for neighbor in adjacency[todo.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    assert len(seen) == len(points), "Disconnected cloth"
    return {"vertices": len(points), "triangles": len(faces), "area_m2": float(areas.sum()), "max_edge_multiplicity": max(edges.values()), "connected": True}


def surface_edges(faces):
    """Return unique material edges for repeated, read-only strain measurement."""
    return np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1), axis=0)


def edge_strain(rest, points, faces, *, edges=None):
    """Return relative changes in edge length without modifying simulation."""
    if edges is None:
        edges = surface_edges(faces)
    a, b = edges.T
    return np.linalg.norm(points[b] - points[a], axis=1) / np.linalg.norm(rest[b] - rest[a], axis=1) - 1


def split_lip(points, faces, markers, lip_width=0.010):
    """Partition the textile and the finite rubber mouth strip at a sewn edge.

    Return local meshes and the shared original vertex indices. This models a
    sewn-in rubber border, without attaching anything to the holder or world.
    """
    base = set(markers["base"])
    depth = markers["dimensions_m"][2]
    is_lip = []
    for face in faces:
        upper = any(int(i) not in base for i in face)
        is_lip.append(upper and points[face, 0].mean() > depth - lip_width + 1e-8)
    parts = []
    for mask in (np.logical_not(is_lip), np.asarray(is_lip)):
        subset = faces[mask]
        ids = np.unique(subset)
        lookup = {int(old): new for new, old in enumerate(ids)}
        local_faces = np.array([[lookup[int(i)] for i in f] for f in subset], dtype=np.int32)
        parts.append((ids, points[ids], local_faces))
    shared = np.intersect1d(parts[0][0], parts[1][0])
    return parts, shared


def extrude_surface(points, faces, thickness):
    """Return a conforming three-layer tetrahedral extrusion and middle indices."""
    normals = np.zeros_like(points)
    triangle_normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                points[faces[:, 2]] - points[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], triangle_normals)
    lengths = np.linalg.norm(normals, axis=1)
    assert np.all(lengths > 1e-12), "Undefined extrusion normal"
    normals /= lengths[:, None]
    nodes = np.concatenate([points - thickness * normals / 2, points,
                            points + thickness * normals / 2])
    count = len(points)
    elements = []
    for layer in range(2):
        for a, b, c in np.sort(faces, axis=1):
            a, b, c = (int(v) + layer * count for v in (a, b, c))
            A, B, C = a + count, b + count, c + count
            elements.extend([(a, b, c, C), (a, b, B, C), (a, A, B, C)])
    elements = np.asarray(elements, dtype=np.int32)
    corners = nodes[elements]
    determinants = np.linalg.det(corners[:, 1:] - corners[:, :1])
    assert np.all(np.abs(determinants) > 1e-14), "Degenerate extrusion"
    negative = determinants < 0
    elements[negative, :2] = elements[negative, :2][:, ::-1]
    return nodes, elements, np.arange(count, 2 * count)


def tet_boundary(elements):
    """Return oriented exterior triangles, rejecting nonmanifold tetrahedra."""
    entries = {}
    for a, b, c, d in elements:
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(map(int, face)))
            entries.setdefault(key, []).append(face)
    assert all(len(faces) <= 2 for faces in entries.values()), "Nonmanifold tetrahedra"
    return np.asarray([faces[0] for faces in entries.values() if len(faces) == 1], dtype=np.int32)


def material_rest(points, flat_panels=False, seam_height=0.):
    """Return horizontal flat-panel metric geometry for an explicitly selected variant.

    The sewn panels share boundary vertices but have separate triangle faces.
    Their opposite winding retains the folded seam rest angle. Initial arching
    is a placement deformation; it is not a molded stress-free textile shape.
    This helper is for this task's horizontal setup, not arbitrary posed meshes.
    """
    rest=np.asarray(points).copy()
    if flat_panels:
        floor=rest[:,2].min()
        rest[:,2]=np.where(rest[:,2]>floor+1e-8,floor+seam_height,floor)
    return rest


def sewn_bend_pairs(points, faces):
    """Identify opposite-winding sewn-panel neighbors in the initial pocket."""
    normals=np.cross(points[faces[:,1]]-points[faces[:,0]],
                     points[faces[:,2]]-points[faces[:,0]])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    incident={}
    for i,tri in enumerate(faces):
        for a,b in zip(tri,np.roll(tri,-1)):
            incident.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    pairs=[]
    for ids in incident.values():
        if len(ids)==2 and normals[ids[0]]@normals[ids[1]]<0:
            pairs.append(ids)
    return np.asarray(pairs,dtype=np.int32).reshape(-1,2)
