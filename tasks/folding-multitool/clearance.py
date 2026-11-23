"""Conservative payload clearance from actual tool and stock robot body poses."""
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from yam_arena import YAM_URDF
from yam_arena.urdf import rpy_matrix, quat_xyzw_to_matrix
from geometry import tool_links
from mechanism import load_spec


def corners(points):
    lo, hi = points.min(0), points.max(0)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])])


@lru_cache(maxsize=1)
def local_boxes():
    robot, tool = {}, {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        points = []
        for collision in link.findall("collision"):
            origin, mesh = collision.find("origin"), collision.find("geometry/mesh")
            if mesh is None:
                raise RuntimeError("Unhandled stock collision geometry")
            p = np.asarray(trimesh.load(YAM_URDF.parent/mesh.get("filename"), process=False).vertices)
            p *= np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.]*3
            points.append(p@rpy_matrix(rpy).T+xyz)
        if points:
            robot[link.get("name")] = corners(np.concatenate(points))
    for name, pieces in tool_links(load_spec()).items():
        tool[name] = corners(np.concatenate([p.vertices for p in pieces]))
    return robot, tool


def distance_lower_bound(low, high, tool_poses, robot_poses):
    robot, tool = local_boxes()
    distances = []
    for boxes, poses in [(tool, tool_poses), *[(robot, p) for p in robot_poses]]:
        if not set(boxes).issubset(poses):
            raise RuntimeError("Missing actual body poses for stock collision clearance")
        for name, points in boxes.items():
            pose = np.asarray(poses[name])
            world = points@quat_xyzw_to_matrix(pose[3:]).T+pose[:3]
            separation = np.maximum(np.maximum(np.asarray(low)-world.max(0), world.min(0)-high), 0.)
            distances.append(np.linalg.norm(separation))
    return float(min(distances))
