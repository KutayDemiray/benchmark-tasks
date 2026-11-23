"""Measure pocket-normal margins in recorded states without changing the scorer."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def diagnose(run, start, end):
    """Report the existing orientation ratio at each selected control boundary."""
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    hz = json.loads((run / 'request.json').read_text())['physics_hz']
    cover = json.loads((run / 'cover-mesh.json').read_text())
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    upper_faces = []
    for pocket in cover['pockets']:
        base = set(pocket['markers']['base'])
        upper_faces.append(np.asarray([face for face in pocket['faces']
                                      if any(int(vertex) not in base for vertex in face)]))
    records = []
    hashes = {}
    loaded = None
    for row in rows:
        if not start <= row['time_s'] <= end:
            continue
        index = (row['step'] + 1) * (hz // 20) - 1
        if index // hz != loaded:
            loaded = index // hz
            path = run / 'physics-chunks' / f'{loaded:05d}.npz'
            hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as chunk:
                positions = chunk['positions']
                poses = chunk['frame_body_poses']
        points = positions[index % hz]
        pockets = []
        for side, body in enumerate(('Left', 'Right')):
            pose = poses[index % hz, names.index(body)]
            normal = Rotation.from_quat(pose[3:]).as_matrix()[:, 2]
            triangles = points[upper_faces[side]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            areas = np.linalg.norm(normals, axis=1)
            projection = normals @ normal
            ratio = float(projection.sum() / areas.sum())
            pockets.append(dict(orientation_ratio=ratio, margin_above_threshold=ratio - .8,
                                upper_area_m2=float(areas.sum() / 2),
                                projected_area_m2=float(projection.sum() / 2),
                                reversed_area_fraction=float(areas[projection < 0].sum() / areas.sum())))
            assert (ratio > .8) == row['pockets'][side]['orientation_correct']
        records.append(dict(time_s=row['time_s'], phase=row['phase'], pockets=pockets))
    assert records, 'No recorded states in the selected interval'
    minima = []
    for side in range(2):
        worst = min(records, key=lambda item: item['pockets'][side]['orientation_ratio'])
        minima.append(dict(time_s=worst['time_s'], **worst['pockets'][side]))
    for name in ('steps.jsonl', 'request.json', 'cover-mesh.json', 'frame-names.json'):
        hashes[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    report = dict(source_run=run.name, interval_s=[start, end], minima=minima, records=records,
                  scope='Existing orientation metric at recorded control boundaries only; no score or physics certificate.',
                  input_sha256=hashes, diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run / 'pocket-orientation-diagnostic.json').write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key not in ('records', 'input_sha256')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--start', type=float, default=82.)
    parser.add_argument('--end', type=float, default=95.)
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    print(json.dumps(diagnose(args.run, args.start, args.end), indent=2))
