"""Render measured passive-contact states; no physics is executed or inferred."""

from task_paths import OUTPUT_ROOT
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from audit_mechanism import capsule_proxy
from passive_contact import native_physics_errors


def load(run):
    request = json.loads((run/'request.json').read_text())
    assert request['experiment'] == 'passive_contact'
    assert json.loads((run/'result.json').read_text())['status'] == 'completed'
    assert not native_physics_errors((run/'kit.log').read_text()), 'Cannot render rejected native physics as a comparison'
    assert not (run/'native-constraint-rejection.json').exists()
    hz = request['physics_hz']; poses = []; nodes = []; inputs = {}
    for path in sorted((run/'physics-chunks').glob('*.npz')):
        with np.load(path) as chunk:
            indices = np.arange(hz//20-1, len(chunk['frame_body_poses']), hz//20)
            poses.extend(chunk['frame_body_poses'][indices])
            if 'positions' in chunk:
                nodes.extend(chunk['positions'][indices])
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = json.loads((run/'frame-geometry.json').read_text())
    names = json.loads((run/'frame-names.json').read_text())['body_names']
    return dict(poses=np.array(poses), nodes=np.array(nodes), shapes=metadata['colliders'], names=names, inputs=inputs, request=request)


def render(first, second, output, *, fixed_tree=False, solver_refinement=False, contact_refinement=False, sdk_comparison=False):
    runs = [load(first), load(second)]
    frames = min(len(run['poses']) for run in runs)
    assert frames == round(runs[0]['request']['seconds']*20) == round(runs[1]['request']['seconds']*20)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100, facecolor='#f5f7fa')
    fig.subplots_adjust(top=.78, bottom=.2, left=.07, right=.98, wspace=.18)
    if fixed_tree:
        assert runs[0]['request'].get('passive_freeze_joints') == 'all'
        assert runs[1]['request']['passive_mode'] == 'compound'
    assert sum((fixed_tree, solver_refinement, contact_refinement, sdk_comparison)) <= 1
    if solver_refinement:
        assert all(run['request']['passive_mode'] == 'articulated' and run['request']['passive_solver'] == 'PGS' for run in runs)
        assert runs[1]['request']['physics_hz'] == 2*runs[0]['request']['physics_hz']
    if contact_refinement:
        assert all(run['request']['passive_mode'] == 'articulated' and run['request'].get('passive_head_correction') for run in runs)
        assert runs[1]['request']['physics_hz'] == 2*runs[0]['request']['physics_hz']
    if sdk_comparison:
        assert not runs[0]['request'].get('passive_sdk_scene')
        assert runs[1]['request'].get('passive_sdk_scene') == first.name
        assert runs[0]['request']['physics_hz'] == runs[1]['request']['physics_hz']
    bounded_match = sdk_comparison and runs[1]['request'].get('passive_cloth_correction') == 'bounded'
    if bounded_match:
        proof = json.loads((second/'native-default-verification.json').read_text())
        assert proof['scene_identity_verified'] and proof['native_cloth_correction_m_s'] == 3.
    title = ('Same cloth correction limit; frame response still differs' if bounded_match else
        'Same authored scene: compare installed PhysX runtimes' if sdk_comparison else
        'Head-contact correction: timestep check fails' if contact_refinement else
        'Solver refinement: the apparent fix does not hold' if solver_refinement else
        'Same rigid assembly: multi-link versus one body' if fixed_tree else 'Passive contact: isolate the moving joints')
    fig.suptitle(title, fontsize=19, y=.96)
    subtitle = ('Native cloth correction: 3 m/s in both runtimes. Other native defaults still differ.' if bounded_match else
        'Same recorded start, cloth material and external frame geometry; no robots or controller')
    fig.text(.5, .89, subtitle, ha='center', fontsize=11)
    fig.text(.5, .055, 'Measured native-state visualization at actual speed. Initialized diagnostic; not a full-task demonstration.', ha='center', fontsize=10)
    titles = (['Seven fixed links; no moving joints', 'One equivalent rigid body'] if fixed_tree else
        ['Articulated frame', 'Diagnostic control: joints frozen into one body'])
    if solver_refinement:
        titles = [f'PGS: {run["request"]["physics_hz"]} physics steps/s' for run in runs]
    if contact_refinement:
        titles = [f'Head correction 0.1 m/s: {run["request"]["physics_hz"]} physics steps/s' for run in runs]
    if sdk_comparison:
        titles = ['Kit PhysX: released frame overturns', 'Standalone SDK: level; cloth settling still fails']
    panels = []
    colors = dict(Left='#8a929d', Right='#8a929d', Bridge='#5d6673', Handle='#4e5968', Rocker='#d67242', Clip='#dfad25')
    for run, ax, title in zip(runs, axes, titles):
        ax.set(xlim=(-.65, .22), ylim=(-.025, .36), aspect='equal', xlabel='Table width direction (m)', ylabel='Height (m)')
        ax.set_title(title, fontsize=12)
        ax.axhline(0, color='#646b73', lw=2); ax.grid(alpha=.15)
        patches = []
        for shape in run['shapes']:
            if not shape['enabled']:
                continue
            vertices = capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
            patch = Polygon(np.zeros((3, 2)), facecolor=colors.get(shape['body'], '#929ca8'), edgecolor='#303945', lw=.4)
            ax.add_patch(patch); patches.append((patch, vertices, run['names'].index(shape['body'])))
        textile = ax.scatter([], [], s=.6, color='#2379d1', alpha=.32, linewidths=0, zorder=4)
        caption = ax.text(.03, .95, '', transform=ax.transAxes, va='top', fontsize=11)
        panels.append((patches, textile, caption))
    stamp = fig.text(.5, .12, '', ha='center', fontsize=12)
    writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    assert not output.exists(), 'Preserve previous videos'
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output), dpi=100):
        for index in range(frames):
            for run, (patches, textile, caption) in zip(runs, panels):
                for patch, points, body in patches:
                    pose = run['poses'][index, body]
                    world = Rotation.from_quat(pose[3:]).apply(points)+pose[:3]
                    projection = world[:, [1, 2]]
                    patch.set_xy(projection[ConvexHull(projection).vertices])
                    patch.set_zorder(2-float(world[:, 0].mean()))
                if len(run['nodes']):
                    textile.set_offsets(run['nodes'][index][:, [1, 2]])
                normal = Rotation.from_quat(run['poses'][index, 0, 3:]).apply([0., 0., 1.])
                angle = np.rad2deg(np.arccos(np.clip(normal[2], -1, 1)))
                caption.set_text(f'Head inclination {angle:.2f}°')
            stamp.set_text(f'{(index+1)/20:.2f} / {frames/20:.2f} s   •   blue: textile   •   gray: frame')
            if index == 19:
                fig.savefig(output.with_suffix('.preview.png'))
            writer.grab_frame()
    plt.close(fig)
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-', '-progress', 'pipe:1', '-nostats'], capture_output=True, text=True)
    counts = [int(line.split('=', 1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    assert decoded.returncode == 0 and counts[-1] == frames, decoded.stderr
    proof = dict(diagnostic_only=True, benchmark_eligible=False, kind='measured_native_geometry_visualization',
        sources=[first.name, second.name], matched_cloth_correction=bounded_match, fixed_tree_comparison=fixed_tree, solver_refinement_comparison=solver_refinement, contact_refinement_comparison=contact_refinement, sdk_comparison=sdk_comparison, decoded_frames=frames, duration_s=frames/20, playback_speed=1,
        source_sha256={key: value for run in runs for key, value in run['inputs'].items()},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(proof, indent=2))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path); parser.add_argument('second', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--fixed-tree', action='store_true')
    parser.add_argument('--solver-refinement', action='store_true')
    parser.add_argument('--contact-refinement', action='store_true')
    parser.add_argument('--sdk-comparison', action='store_true')
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(OUTPUT_ROOT)
    render(args.first, args.second, args.output, fixed_tree=args.fixed_tree, solver_refinement=args.solver_refinement, contact_refinement=args.contact_refinement, sdk_comparison=args.sdk_comparison)
