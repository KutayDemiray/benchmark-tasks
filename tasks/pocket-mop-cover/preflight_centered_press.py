"""Screen accessible crossbar press points using the recorded pre-press scene."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from audit_frame_robot import compare, prepare
from outcome_native import cloth_clearance, shape_clearance
from preflight_cover import KinematicSession
from preflight_supported_release import projection_shape
from preflight_upright_release import moved
from robot_clearance import link_transforms
from robot_reopen import transform


def preflight(run, out, quarter_turn=False):
    """Compare approach and touch kinematics; do not certify pressing dynamics."""
    from station_coordinates import TRANSLATION_S_W

    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines() if line.endswith('}')]
    index = max(i for i, row in enumerate(rows) if row['phase'] == 'press_frame_flat')
    row = rows[index]
    hz = json.loads((run / 'request.json').read_text())['physics_hz']
    native_index = (index + 1) * (hz // 20) - 1
    native_file = run / 'physics-chunks' / f'{native_index // hz:05d}.npz'
    with np.load(native_file) as chunk:
        state = {name: chunk[name][native_index % hz].copy() for name in
                 ('frame_body_poses', 'left_body_poses', 'right_body_poses', 'positions')}
    with np.load(run / 'geometry.npz') as mesh:
        faces = mesh['faces']
    names = json.loads((run / 'frame-names.json').read_text())['body_names']
    robots = json.loads((run / 'robot-cooked-colliders.json').read_text())
    scene = json.loads((run / 'scene-audit.json').read_text())
    frame = json.loads((run / 'frame-geometry.json').read_text())['colliders']
    bridge = transform(state['frame_body_poses'][names.index('Bridge')])
    identity = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    shapes = [prepare(shape, identity) for shape in robots['left']['colliders']]
    hardware = [prepare(shape, state['frame_body_poses'][None, names.index(shape['body'])])
                for shape in frame if shape['enabled']]
    right = [prepare(shape, state['right_body_poses'][None, robots['right']['body_names'].index(shape['body'])])
             for shape in robots['right']['colliders']]
    points = np.array(list(itertools.product((-1., 1.), repeat=3))) * np.array(scene['table_size_m']) / 2
    table = prepare(dict(name='Table', body='Table', points=points),
                    np.array([[*scene['table_center_m'], 0., 0., 0., 1.]]))
    wings = [projection_shape(shape) for shape in hardware if shape['body'] in ('Left', 'Right')]
    action = np.array(row['robot_action'])
    initial = link_transforms(action[:7], 'left', TRANSLATION_S_W)
    residuals = {}
    for name in robots['left']['body_names']:
        actual = transform(state['left_body_poses'][robots['left']['body_names'].index(name)])
        residuals[name] = np.linalg.inv(initial[name]) @ actual

    class Session(KinematicSession):
        def step(self, command, phase):
            super().step(command, phase)
            self.samples.append((command.copy(), phase))

    reports = []
    for index, y in enumerate((.025, .020, .030)):
        folder = out / f'candidate-{index}'
        folder.mkdir()
        session = Session(action, TRANSLATION_S_W, folder, None)
        session.samples = []
        feature = np.array([.030, y, .035])
        point = bridge[:3, 3] + bridge[:3, :3] @ feature + [0., 0., .014]
        rotation = np.diag([-1., 1., -1.])
        if quarter_turn:
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) @ rotation
        report = dict(feature_B_m=feature.tolist(), approach_height_m=.06, source_time_s=row['time_s'], quarter_turn=quarter_turn)
        try:
            session.move('left', point + [0., 0., .06], rotation, 0., 2., 'approach_bridge_press', joint=True)
            session.move('left', point, rotation, 0., 1.5, 'touch_bridge_crossbar')
        except AssertionError as error:
            report.update(passed=False, reach_error=str(error))
            reports.append(report)
            (folder / 'result.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            continue
        models = []
        for recorded_residual in (False, True):
            hits = []
            wing_gap, cloth_gap = float('inf'), float('inf')
            # A colliding destination rejects a candidate without computing
            # its entire approach. Passing candidates still check every pose.
            order = [len(session.samples) - 1, *range(len(session.samples) - 1)]
            for sample in order:
                command, phase = session.samples[sample]
                links = link_transforms(command[:7], 'left', session.shift)
                left = []
                for shape in shapes:
                    residual = residuals[shape['body']] if recorded_residual else np.eye(4)
                    left.append(moved(shape, links[shape['body']] @ residual))
                projection = [projection_shape(shape) for shape in left]
                wing_gap = min(wing_gap, shape_clearance(wings, projection, .002))
                cloth_gap = min(cloth_gap, cloth_clearance(state['positions'][None], faces, projection))
                second = hardware + right + [table]
                low_a, high_a = (np.array([shape[key] for shape in left]) for key in ('global_low', 'global_high'))
                low_b, high_b = (np.array([shape[key] for shape in second]) for key in ('global_low', 'global_high'))
                overlap = np.all(high_a[:, None] >= low_b, axis=2) & np.all(high_b >= low_a[:, None], axis=2)
                for a, b in np.argwhere(overlap):
                    one, two = left[a], second[b]
                    if one['body'] == 'base' and two['body'] == 'Table':
                        continue
                    depth, _, _ = compare(one, two)
                    if depth > .001:
                        hits.append(dict(sample=sample, phase=phase, first=one['name'],
                                         second=two['body'] + '/' + two['name'], depth_bound_m=depth))
                if hits:
                    break
            models.append(dict(recorded_deflection=recorded_residual, hits=hits,
                               minimum_wing_gap_m=wing_gap, minimum_cloth_gap_m=cloth_gap,
                               passed=not hits and wing_gap >= .002 and cloth_gap >= .002))
        report.update(models=models, passed=all(model['passed'] for model in models))
        reports.append(report)
        (folder / 'result.json').write_text(json.dumps(report, indent=2))
        print(index, feature.tolist(), [(model['passed'], len(model['hits']), model['hits'][:2]) for model in models], flush=True)
    report = dict(candidates=reports, source_run=run.name,
                  scope='Only approach and touch kinematics against fixed native geometry. Zero and constant recorded-deflection models. Press force, snap dynamics and cloth response are not certified.',
                  input_sha256={str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (native_file, run / 'robot-cooked-colliders.json', run / 'frame-geometry.json')},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--quarter-turn', action='store_true')
    args = parser.parse_args()
    root = OUTPUT_ROOT
    assert args.run.resolve().parent == args.out.resolve().parent == root
    args.out.mkdir(exist_ok=False)
    preflight(args.run, args.out, args.quarter_turn)
