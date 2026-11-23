"""Measure archived cloth capture and convex support overlap without running simulation."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from yam_arena.analysis import fk_link_poses_world, pose7_to_matrix
from yam_arena.constants import ARM_JOINTS, LEFT_BASE_POS

OUTPUT = OUTPUT_ROOT


def overlap_radius(planes, transform, lo, hi):
    """Return signed largest-ball radius in the intersection of a convex hull and box."""
    planes = np.asarray(planes, dtype=float)
    normals = planes[:, :3] @ transform[:3, :3].T
    offsets = planes[:, 3] + normals @ transform[:3, 3]
    normals = np.vstack([normals, np.eye(3), -np.eye(3)])
    offsets = np.r_[offsets, hi, -np.asarray(lo)]
    norms = np.linalg.norm(normals, axis=1)
    fit = linprog([0., 0., 0., -1.], A_ub=np.c_[normals, norms], b_ub=offsets,
                  bounds=[(None, None)]*4, method='highs')
    assert fit.success, fit.message
    return float(fit.x[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--times', type=float, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.episode.resolve().is_relative_to(OUTPUT)
    assert args.out.resolve().is_relative_to(OUTPUT)
    rows = [json.loads(s) for s in (args.episode/'trace.jsonl').read_text().splitlines()]
    meta = json.loads((args.episode/'topology.json').read_text())
    h, x = meta['support_top_m'], meta['support_front_x_m']
    width, thickness = meta.get('support_width_m', .35), meta.get('support_thickness_m', .04)
    supports = {'plate': ([x, .12-width/2, h-thickness], [x+.4, .12+width/2, h])}
    if width < .12:
        supports['post'] = ([x+.08, .10, 0.], [x+.12, .14, h-thickness])
    initial = np.asarray(meta['cloth']['upper']['initial'])
    patch = (initial[:, 0] < initial[:, 0].min()+.005) & (np.abs(initial[:, 1]-.12) < .0061)
    arrays = np.load(args.episode/'control-steps.npz')
    records = []
    for requested in args.times:
        index = min(range(len(rows)), key=lambda i: abs(rows[i]['time_s']-requested))
        row = rows[index]
        actual = {name: pose7_to_matrix(pose) for name, pose in
                  zip(row['left_link_names'], row['left_link_poses_xyzw'])}
        action = row['robot_action']
        joints = dict(zip(ARM_JOINTS, action[:6]))
        joints.update(joint7=-.04695*action[6], joint8=-.04695*action[6])
        command = fk_link_poses_world(joints, LEFT_BASE_POS)
        nodes = arrays['cloth_upper'][index]
        local = (nodes[patch]-np.asarray(row['measured_tcp_m'])) @ actual['gripper'][:3, :3]
        overlaps = []
        for kind, poses in (('measured', actual), ('commanded', command)):
            for support, (lo, hi) in supports.items():
                best, link = -float('inf'), None
                for name, parts in meta['robot_collision_planes'].items():
                    if name not in poses:
                        continue
                    for planes in parts:
                        radius = overlap_radius(planes, poses[name], lo, hi)
                        if radius > best:
                            best, link = radius, name
                overlaps.append({'pose_kind': kind, 'support': support,
                                 'maximum_overlap_ball_radius_m': best, 'link': link})
        records.append({'time_s': row['time_s'], 'phase': row['control']['phase'],
                        'tcp_target_error_m': float(np.linalg.norm(np.asarray(row['measured_tcp_m'])-row['control']['target_tcp_m'])),
                        'actual_half_gap_m': row['joint_positions'][6]*.04695,
                        'patch_local_min_m': local.min(axis=0).tolist(),
                        'patch_local_max_m': local.max(axis=0).tolist(),
                        'patch_world_center_m': nodes[patch].mean(axis=0).tolist(),
                        'support_intersections': overlaps})
    result = {'episode': str(args.episode), 'records': records,
              'trace_sha256': hashlib.sha256((args.episode/'trace.jsonl').read_bytes()).hexdigest(),
              'method': 'All recorded convex hull planes versus support AABBs. Positive radius proves solid overlap in the pose geometry; negative is a separating ball margin, not Euclidean distance. Patch coordinates use actual gripper frame and TCP. Recorded states only; no simulation or gate promotion.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
