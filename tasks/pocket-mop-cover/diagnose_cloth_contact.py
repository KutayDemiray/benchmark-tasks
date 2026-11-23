"""Inspect one cloth/frame contact in completed native recording chunks."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from robot_contacts import triangles_in_convex


def digest(path):
    """Hash an archived input without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(run, body, shape_name, start, end):
    """Check every substep in (start, end] seconds against one frame collider."""
    assert 0 <= start < end
    request = json.loads((run / 'request.json').read_text())
    assert request['robot_cover']
    hz = request['physics_hz']
    kernel = Path(__file__).with_name('robot_contacts.py')
    assert digest(kernel) == digest(run / 'source/robot_contacts.py'), 'Contact kernel differs from recorded source'
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    colliders = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    selected = [item for item in colliders if item['body'] == body and item['name'] == shape_name]
    assert len(selected) == 1 and selected[0]['enabled'] and 'points' in selected[0]
    vertices = np.asarray(selected[0]['points'])
    planes = ConvexHull(vertices).equations
    low, high = vertices.min(0), vertices.max(0)
    body_index = names.index(body)
    with np.load(run / 'geometry.npz') as geometry:
        faces = geometry['faces']
    hashes = {name: digest(run / name) for name in (
        'request.json', 'frame-names.json', 'frame-geometry.json', 'geometry.npz', 'manifest.json')}
    violations = []
    checked = 0
    for second in range(start, end):
        path = run / 'physics-chunks' / f'{second:05d}.npz'
        assert path.is_file(), f'Native chunk is not available: {path.name}'
        hashes[str(path.relative_to(run))] = digest(path)
        with np.load(path) as chunk:
            points = chunk['positions']
            poses = chunk['frame_body_poses'][:, body_index]
        assert len(points) == len(poses) == hz
        assert np.isfinite(points).all() and np.isfinite(poses).all()
        for index, (world, pose) in enumerate(zip(points, poses)):
            local = (world - pose[:3]) @ Rotation.from_quat(pose[3:]).as_matrix()
            triangles = local[faces]
            nearby = np.all(triangles.max(1) >= low, axis=1) & np.all(triangles.min(1) <= high, axis=1)
            candidates = faces[nearby]
            checked += 1
            if not len(candidates) or not triangles_in_convex(local, candidates, planes):
                continue
            lower, upper = .001, float(np.linalg.norm(high - low))
            while upper - lower > 1e-6:
                middle = (lower + upper) / 2
                if triangles_in_convex(local, candidates, planes, middle):
                    lower = middle
                else:
                    upper = middle
            step = second * hz + index
            violations.append(dict(physics_step=step, time_s=(step + 1) / hz,
                triangle_interior_depth_bracket_m=[lower, upper]))
    report = dict(source_run=run.name, diagnostic_only=True, complete_episode_certificate=False,
        scope='Every saved substep in the selected interval against one frame collider. Other contacts, strain, self-collision and the rest of the episode require the complete audits. Depth is triangle entry into the convex interior, not a contact-force measurement.',
        time_interval_s=[start, end], interval_convention='(start, end]',
        collider=f'Frame/{body}/{shape_name}', physics_steps=checked,
        tolerance_m=.001, violating_substeps=violations,
        selected_contact_check_passed=not violations, source_sha256=hashes,
        contact_kernel_sha256=digest(kernel), diagnostic_source_sha256=digest(Path(__file__)))
    (run / 'focused-cloth-contact-diagnostic.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--body', required=True)
    parser.add_argument('--shape', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    result = diagnose(args.run, args.body, args.shape, args.start, args.end)
    result.pop('source_sha256')
    print(json.dumps(result, indent=2))
