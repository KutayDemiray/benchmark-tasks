"""Independent geometric diagnostics, without simulator state writes."""

import numpy as np


def closed_pocket_faces(rest, faces, markers):
    """Close only the assessor's mathematical pocket volume at its open mouth."""
    base = set(markers["base"])
    depth = markers["dimensions_m"][2]
    selected = []
    for f in faces:
        if any(int(i) not in base for i in f) or rest[f, 0].mean() < depth - 1e-9:
            selected.append(f.tolist())
    upper = markers["upper_grid"][-1]
    lower = markers["base_grid"][len(markers["upper_grid"]) - 1]
    for j in range(len(upper) - 1):
        a, b, c, d = lower[j], lower[j + 1], upper[j + 1], upper[j]
        for tri in ((a, b, c), (a, c, d)):
            if len(set(tri)) == 3:
                selected.append(tri)
    return np.asarray(selected, dtype=int)


def inside(points, vertices, closed_faces):
    """Use signed solid angle of the deformed closed pocket, not proximity."""
    triangle = vertices[closed_faces]
    result = []
    for point in np.asarray(points):
        a, b, c = np.moveaxis(triangle - point, 1, 0)
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb
        winding = np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi)
        result.append(abs(winding) > 0.9)
    return np.asarray(result)


def triangles_in_box(vertices, faces, center, size, tolerance=0.001):
    """SAT test catches face/edge passage even when all mesh nodes miss a box.

    Return the count of triangles intersecting the box eroded by tolerance.
    Diagnostic blade orientation is identity; callers must transform other
    boxes into their local frame before using this function.
    """
    half = np.asarray(size) / 2 - tolerance
    assert np.all(half > 0)
    t = vertices[faces] - center
    valid = np.all(t.min(1) < half, axis=1) & np.all(t.max(1) > -half, axis=1)
    candidates = t[valid]
    if not len(candidates):
        return 0
    edges = [candidates[:, 1] - candidates[:, 0], candidates[:, 2] - candidates[:, 1], candidates[:, 0] - candidates[:, 2]]
    axes = [np.cross(edges[0], edges[1])]
    for edge in edges:
        for axis in np.eye(3):
            axes.append(np.cross(edge, axis))
    hit = np.ones(len(candidates), dtype=bool)
    for axis in axes:
        projection = np.einsum("ijk,ik->ij", candidates, axis)
        radius = np.abs(axis) @ half
        nondegenerate = np.linalg.norm(axis, axis=1) > 1e-12
        separated = (projection.min(1) > radius) | (projection.max(1) < -radius)
        hit &= ~(nondegenerate & separated)
    return int(hit.sum())


def pocket_seating(vertices, closed_faces, center, size, *, markers=None, faces=None, profile="square", rotation=None):
    """Check 40 mm of the leading end across 80% of its 100 mm width."""
    from blade_geometry import sample_height
    rotation=np.eye(3) if rotation is None else np.asarray(rotation)
    samples = []
    for x in (0.002, 0.02, 0.04):
        for y in np.linspace(-0.4 * size[1], 0.4 * size[1], 9):
            samples.append(center + rotation @ np.array([x - size[0] / 2, y, sample_height(x,size,profile)]))
    containment = inside(samples, vertices, closed_faces).reshape(3, 9)
    upright = True
    if markers is not None:
        assert faces is not None
        base = set(markers["base"])
        upper = []
        for f in faces:
            if any(int(i) not in base for i in f):
                upper.append(f)
        triangles = vertices[np.asarray(upper)]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        # Correct installation is relative to the frame end, including when
        # the assembled mop tilts during the functional retention check.
        upright = bool((normals @ rotation[:, 2]).sum() / np.linalg.norm(normals, axis=1).sum() > 0.8)
    return {"seated_samples": bool(containment.all() and upright), "inside_fraction": float(containment.mean()), "orientation_correct": upright}
