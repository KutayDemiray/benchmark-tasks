"""Summarize native assembly phases and plot measured cover/frame sections."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from robot_reopen import transform


def inspect(run):
    """Write diagnostic geometry without changing or certifying an episode."""
    from yam_arena import ik, station
    from station_coordinates import TRANSLATION_S_W

    request = json.loads((run / 'request.json').read_text())
    names = json.loads((run / 'frame-names.json').read_text())
    shapes = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    scene = json.loads((run / 'scene-audit.json').read_text())
    cover = json.loads((run / 'cover-mesh.json').read_text())
    initial_nodes = np.load(run / 'geometry.npz')['initial']
    middle = cover['middle_join_node_ids']
    shift = TRANSLATION_S_W if request['table_origin'] else np.zeros(3)
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    chunks = sorted((run / 'physics-chunks').glob('[0-9][0-9][0-9][0-9][0-9].npz'))
    finished = (run / 'result.json').exists()
    last_safe = len(rows) if finished else min(len(rows), max(0, len(chunks) - 1) * 20)
    rows = rows[:last_safe]
    selected = {}
    for index, row in enumerate(rows):
        selected[row['phase']] = index
    reports, frames = [], []
    initial = None
    for phase, index in selected.items():
        if index < 35 * 20:
            continue
        physics_index = (index + 1) * (request['physics_hz'] // 20) - 1
        with np.load(chunks[physics_index // request['physics_hz']]) as chunk:
            offset = physics_index % request['physics_hz']
            poses = chunk['frame_body_poses'][offset]
            cloth = chunk['positions'][offset]
        row = rows[index]
        bridge = transform(poses[names['body_names'].index('Bridge')])
        local = {}
        for side, offset, body in (('left', 0, 'Left'), ('right', 7, 'Handle')):
            position, rotation = ik.tcp_pose(np.array(row['measured_joint_pos'])[offset:offset + 6], station.BASES[side])
            wrist = np.eye(4)
            wrist[:3, 3], wrist[:3, :3] = position + shift, rotation
            owner = transform(poses[names['body_names'].index(body)])
            local[side] = np.linalg.inv(owner) @ wrist
        if phase == 'verify_inner_rail_grasp':
            initial = local
        slip = {} if initial is None else {
            side: float(np.linalg.norm(local[side][:3, 3] - initial[side][:3, 3]))
            for side in local}
        tips = []
        for body, sign in (('Left', -1.), ('Right', 1.)):
            pose = transform(poses[names['body_names'].index(body)])
            tips.append((pose[:3, 3] + pose[:3, :3] @ [sign * .192, 0., -.0025]).tolist())
        reports.append(dict(phase=phase, time_s=row['time_s'], wing_angles_deg=row['wing_angles_deg'],
            bridge_xyz_m=bridge[:3, 3].tolist(),
            bridge_rpy_deg=Rotation.from_matrix(bridge[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            tcp_local_m={side: value[:3, 3].tolist() for side, value in local.items()},
            tcp_translation_from_grasp_m=slip, pockets=row['pockets'],
            leading_nose_midpoints_m=tips,
            cover_middle_displacement_m=(cloth[middle].mean(0)-initial_nodes[middle].mean(0)).tolist(),
            pocket_mouth_midpoints_m=[cloth[pocket['markers']['mouth_upper']].mean(0).tolist()
                                      for pocket in cover['pockets']],
            max_abs_edge_strain=row['max_abs_edge_strain']))
        frames.append((row, poses, cloth))
    (run / 'assembly-diagnostic.json').write_text(json.dumps(dict(
        scope='Read-only native phase summaries; not a benchmark score or contact certificate.',
        observed_control_steps=len(rows), phases=reports), indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted = ('verify_inner_rail_grasp', 'align_pocket_mouths', 'hold_guided_frame',
              'hold_pressed_frame', 'hold_flat_frame', 'released')
    frames = [entry for entry in frames if entry[0]['phase'] in wanted]
    if not frames:
        return reports
    fig, axes = plt.subplots(len(frames), 1, figsize=(12, 2.6 * len(frames)), squeeze=False)
    table_top = scene['table_center_m'][2] + scene['table_size_m'][2] / 2
    for axis, (row, poses, cloth) in zip(axes[:, 0], frames):
        # Central material strip, projected into XZ. These are native nodes,
        # not a replacement simulation or a claim that every surface is shown.
        center_y = np.median(cloth[:, 1])
        strip = abs(cloth[:, 1] - center_y) < .004
        axis.scatter(cloth[strip, 0], cloth[strip, 2], s=2, color='#175d9c', label='Cloth central strip')
        for shape in shapes:
            if not shape['enabled'] or shape['body'] not in ('Left', 'Right'):
                continue
            pose = poses[names['body_names'].index(shape['body'])]
            points = Rotation.from_quat(pose[3:]).apply(shape['points']) + pose[:3]
            projected = points[:, [0, 2]]
            hull = ConvexHull(projected)
            axis.fill(projected[hull.vertices, 0], projected[hull.vertices, 1], color='#ed9542', alpha=.5)
        axis.axhline(table_top, color='black', linewidth=.8)
        head_y = poses[names['body_names'].index('Bridge'), 1]
        axis.set_title(f"{row['phase']} — {row['time_s']:.2f} s — wings {row['wing_angles_deg'][1]:.1f}°"
                       f" — head/cloth Y: {head_y:.3f}/{center_y:.3f} m")
        axis.set_ylabel('Native Z (m)')
        axis.set_xlim(-.38, .18)
        axis.set_ylim(-.015, max(.20, float(cloth[:, 2].max()) + .03))
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel('Native X (m); central section is diagnostic, not a contact audit')
    fig.tight_layout()
    fig.savefig(run / 'assembly-sections.png', dpi=160)
    plt.close(fig)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    print(json.dumps(inspect(args.run), indent=2))
