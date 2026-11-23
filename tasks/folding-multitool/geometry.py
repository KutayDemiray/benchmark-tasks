"""Procedural real-envelope compound geometry; no simulator imports."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull

from mechanism import BODIES, link_transforms, reset_joints

SUPPORT_HEIGHT = .055


@dataclass
class Piece:
    name: str
    vertices: np.ndarray
    color: tuple = (0.60, 0.62, 0.64)
    collision: bool = True

    @property
    def triangles(self):
        h = ConvexHull(self.vertices)
        faces = h.simplices.copy()
        for i, face in enumerate(faces):
            a, b, c = self.vertices[face]
            if np.dot(np.cross(b-a, c-a), h.equations[i, :3]) < 0:
                faces[i, [1, 2]] = faces[i, [2, 1]]
        return faces


def box(name, low, high, color=(0.60, 0.62, 0.64), collision=True):
    lo, hi = np.asarray(low), np.asarray(high)
    if np.any(hi <= lo):
        raise ValueError(f"Non-positive box: {name}")
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return Piece(name, pts, color, collision)


def prism(name, xy, zlo, zhi, color=(0.60, 0.62, 0.64), collision=True):
    return Piece(name, np.array([[x, y, z] for z in (zlo, zhi) for x, y in xy]), color, collision)


def cylinder(name, center, radius, height, collision=True):
    center = np.asarray(center)
    pts = np.array([[radius*np.cos(a), radius*np.sin(a), z]
                    for z in (-height/2, height/2) for a in np.linspace(0, 2*np.pi, 24, endpoint=False)])
    return Piece(name, pts+center, (0.48, 0.50, 0.52), collision)


def beam(name, p0, p1, width, zlo, zhi):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    d = p1-p0
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * width/2
    return prism(name, [p0+n, p1+n, p1-n, p0-n], zlo, zhi)


def tool_links(spec):
    g = spec["geometry"]
    # Published thickness includes the two 0.7 mm rivet heads.
    w, h, t = g["handle_width"], g["handle_thickness"]-.0014, g["shell_wall"]
    tail, front = g["handle_tail_x"], g["handle_front_x"]
    a, b = g["fold_pivot_x"], g["fold_pivot_half_span"]
    links = {}
    for label, sign in (("a", 1), ("b", -1)):
        c = (0.49, 0.51, 0.53) if label == "a" else (0.66, 0.67, 0.68)
        # Rounded hinge ears are essential: rectangular tips intersect the other
        # handle during folding. These are the same visible and collision surfaces.
        outline = [(tail, -w/2), (-.012, -w/2)]
        outline += [(front*np.cos(a), front*np.sin(a)) for a in np.linspace(-np.pi/2, np.pi/2, 13)]
        outline += [(-.012, w/2), (tail, w/2)]
        parts = [prism("shell_top", outline, h/2-t, h/2, c),
                 prism("shell_bottom", outline, -h/2, -h/2+t, c),
                 box("tail_cap", (tail, -w/2, -h/2+t), (tail+t, w/2, h/2-t), c)]
        # Inward-facing U channel when folded; the hinge-end cutout clears the jaw tang.
        ys = (w/2-t, w/2) if sign == 1 else (-w/2, -w/2+t)
        parts.append(box("spine", (tail+t, ys[0], -h/2+t), (-.012, ys[1], h/2-t), c))
        parts.append(box("stowed_implements", (tail+t, -w/2+t, -h/2+t),
                         (tail+g["stowed_stack_tail_length"], w/2-t, h/2-t), (0.38, .40, .42)))
        for z in (-h/2-.00035, h/2+.00035):
            parts.append(cylinder(f"pivot_cap_{'lower' if z < 0 else 'upper'}", (0, 0, z), .004, .0007))
        links[f"handle_{label}"] = parts
        jh, gap, tip = g["jaw_thickness"], g["jaw_tip_half_gap"], g["jaw_tip_x"]
        shape = [(0.006, .0006), (tip, gap), (tip, g["jaw_tip_outer_y"]),
                 (.032, .005), (.015, .010), (.006, .010)]
        shape = [(x, sign*y) for x, y in shape]
        # The crossing tangs occupy different axial layers; only the bearing itself is visual.
        z0, z1 = (.00055, .00345) if sign == 1 else (-.00345, -.00055)
        links[f"jaw_{label}"] = [prism("needle_gripping_jaw", shape, -jh/2, jh/2),
                                  beam("crossing_tang", (a, -sign*b), (.010, sign*.005), .006, z0, z1),
                                  cylinder("central_bearing", (0, 0, sign*.0018),
                                           g["central_bearing_radius"], .0035, collision=False)]
    return {name: links[name] for name in BODIES}


def points_at(spec, q, collision_only=False):
    poses = link_transforms(spec, q)
    out = {}
    for name, pieces in tool_links(spec).items():
        tr = poses[name]
        pts = [p.vertices @ tr[:3, :3].T+tr[:3, 3] for p in pieces if p.collision or not collision_only]
        out[name] = np.concatenate(pts)
    return out


def bounds(spec, q):
    pts = np.concatenate(list(points_at(spec, q).values()))
    return pts.min(0), pts.max(0)


def reset_root(spec):
    lo, hi = bounds(spec, reset_joints(spec))
    return np.array([.38-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, .7475+SUPPORT_HEIGHT-lo[2]+.0005])


def mass_properties(pieces, mass):
    """Volume-weighted tetrahedral integrals; inertias exclude no visual metal."""
    volume, first, second = 0., np.zeros(3), np.zeros((3, 3))
    for piece in pieces:
        v = piece.vertices
        center = v.mean(0)
        for face in piece.triangles:
            vertices = np.vstack([center, v[face]])
            dv = abs(np.linalg.det((v[face]-center).T))/6
            vs = vertices.sum(0)
            volume += dv
            first += dv*vs/4
            second += dv*(np.outer(vs, vs)+vertices.T@vertices)/20
    com = first/volume
    covariance = second/volume-np.outer(com, com)
    inertia = mass*(np.trace(covariance)*np.eye(3)-covariance)
    if np.min(np.linalg.eigvalsh(inertia)) <= 0:
        raise ValueError("Non-positive physical inertia")
    return com, inertia


def separating_axis_gap(a, b):
    """Exact SAT overlap for convex polyhedra; positive is a separating projection.

    Negative means penetration on all tested axes. This is not Euclidean clearance.
    """
    ha, hb = ConvexHull(a), ConvexHull(b)
    axes = [*ha.equations[:, :3], *hb.equations[:, :3]]
    def edges(hull, points):
        return [points[j]-points[i] for i, j in {tuple(sorted((int(f[i]), int(f[(i+1)%3]))))
                for f in hull.simplices for i in range(3)}]
    ea, eb = edges(ha, a), edges(hb, b)
    cross = np.cross(np.asarray(ea)[:, None, :], np.asarray(eb)[None, :, :]).reshape(-1, 3)
    lengths = np.linalg.norm(cross, axis=1)
    axes = np.vstack([axes, cross[lengths > 1e-10]/lengths[lengths > 1e-10, None]])
    axes = np.unique(np.round(axes, 9), axis=0)
    pa, pb = a@axes.T, b@axes.T
    return float(np.max(np.maximum(pb.min(0)-pa.max(0), pa.min(0)-pb.max(0))))
