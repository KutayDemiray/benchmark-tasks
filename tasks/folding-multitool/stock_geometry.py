"""Read-only CPU geometry from the installed YAM collision meshes."""
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from yam_arena import YAM_URDF
from yam_arena.urdf import forward_kinematics, rpy_matrix
from yam_arena.constants import FINGER_JOINTS
from mechanism import transform
from geometry import separating_axis_gap


@lru_cache(maxsize=16)
def gripper_meshes(gap, finger_positions=None):
    urdf = ET.parse(YAM_URDF).getroot()
    positions = (-gap/2, -gap/2) if finger_positions is None else finger_positions
    fk = forward_kinematics(YAM_URDF, dict(zip(FINGER_JOINTS, positions)), root="gripper")
    meshes = []
    for name in ("gripper", "tip_left", "tip_right"):
        for i, c in enumerate(urdf.find(f"link[@name='{name}']").findall("collision")):
            origin = c.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            tr = transform([0, 0, -.134])@fk[name]@transform(xyz, rpy_matrix(rpy))
            mesh = c.find("geometry/mesh")
            vertices = trimesh.load(YAM_URDF.parent/mesh.get("filename"), process=False).vertices
            vertices = vertices*np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            meshes.append((f"{name}/{i}", vertices@tr[:3, :3].T+tr[:3, 3]))
    return meshes


def placed(meshes, tr):
    return [(name, vertices@tr[:3, :3].T+tr[:3, 3]) for name, vertices in meshes]


def overlaps(first, second, tolerance=.00005):
    found = []
    for an, av in first:
        for bn, bv in second:
            if np.any(av.max(0) < bv.min(0)) or np.any(bv.max(0) < av.min(0)):
                continue
            gap = separating_axis_gap(av, bv)
            if gap < -tolerance:
                found.append({"a": an, "b": bn, "penetration_mm": float(-1000*gap)})
    return found
