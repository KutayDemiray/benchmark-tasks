"""Observed inner-face pinch predicate for the later load/extraction gates."""
import numpy as np
from yam_arena.urdf import quat_xyzw_to_matrix


def inner_pinch(spec, poses, forces, points):
    """Forces act on the payload; first two partners are jaw A and jaw B.

    Remaining partners must include the handles and robot bodies. Their force
    exclusion is a caller responsibility because stage fixtures vary. Average
    contact points are corroborating evidence, not proof of every contact patch.
    """
    forces, points = np.asarray(forces, float), np.asarray(points, float)
    if forces.ndim != 2 or points.ndim != 2 or forces.shape[0] < 2 or points.shape[0] < 2:
        return False
    if forces.shape[1] != 3 or points.shape[1] != 3:
        return False
    f, p = forces[:2], points[:2]
    if not np.isfinite(f).all() or not np.isfinite(p).all():
        return False
    magnitudes = np.linalg.norm(f, axis=1)
    if np.min(magnitudes) <= .1:
        return False
    directions = f/magnitudes[:, None]
    if np.dot(*directions) > -.85:
        return False
    g = spec["geometry"]
    for i, (body, sign) in enumerate((("jaw_a", 1), ("jaw_b", -1))):
        pose = np.asarray(poses[body])
        rot = quat_xyzw_to_matrix(pose[3:])
        local = rot.T@(p[i]-pose[:3])
        inner_y = .0006+(local[0]-.006)*(g["jaw_tip_half_gap"]-.0006)/(g["jaw_tip_x"]-.006)
        if not g["jaw_tip_x"]-.012 <= local[0] <= g["jaw_tip_x"]+.001:
            return False
        if abs(local[1]-sign*inner_y) > .0015 or abs(local[2]) > g["jaw_thickness"]/2+.001:
            return False
        if np.dot(directions[i], -sign*rot[:, 1]) < .85:
            return False
    return True
