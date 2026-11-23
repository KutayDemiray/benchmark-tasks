"""Offline bundle, trajectory, reset and independent replay audit."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from scoring import analyze


def audit(run):
    setups = list(run.glob('episode-*/setup.json'))
    if setups and json.loads(setups[0].read_text()).get('stage') == 'tabletop_shirt':
        from verify_tabletop_run import verify
        return verify(run)
    if setups and json.loads(setups[0].read_text()).get('stage') == 'garment_diagnostic':
        from verify_garment_run import verify
        return verify(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    hashes = {}
    for name in manifest['sources']:
        hashes[name] = hashlib.sha256((run / 'task-bundle' / name).read_bytes()).hexdigest()
    assert hashes == manifest['sources'], 'Archived bundle was modified'
    assert hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == manifest['bundle_sha256']
    spec = importlib.util.spec_from_file_location('archived_snap_scorer', run / 'task-bundle/scoring.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    episodes = []
    for path in sorted(run.glob('episode-*')):
        if not (path / 'result.json').exists():
            episodes.append({'episode': path.name, 'status': 'incomplete'})
            continue
        setup = json.loads((path / 'setup.json').read_text())
        trace = [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]
        if setup.get('stage') == 'single_snap':
            stored = json.loads((path / 'result.json').read_text())
            robot_path = run / 'task-bundle/robot_scoring.py'
            if not robot_path.exists():
                episodes.append({'episode': path.name, 'source_hash_matches': True,
                                 'independent_rescore_matches': None,
                                 'limitation': 'Early robot integration predates the separate robot scorer',
                                 'full_task_score': stored['full_task_score']})
                continue
            robot_spec = importlib.util.spec_from_file_location('archived_robot_scorer', robot_path)
            robot_scorer = importlib.util.module_from_spec(robot_spec)
            robot_spec.loader.exec_module(robot_scorer)
            physics = [json.loads(line) for line in (path / 'physics.jsonl').read_text().splitlines()]
            topology = json.loads((path / 'topology.json').read_text())
            robot_analyze = robot_scorer.analyze_press_lift if setup['case']=='press_lift' else robot_scorer.analyze_press if setup['case'] in ('press', 'no_press') else robot_scorer.analyze_grasp
            score = robot_analyze(trace, physics, setup['parameters'], topology)
            assert all(score[k] == stored[k] for k in score), f'Robot rescore differs: {path}'
            data = np.load(path / 'control-steps.npz')
            assert np.array_equal(data['actions'], np.array([r['robot_action'] for r in trace]))
            assert all(np.isfinite(data[key]).all() for key in data.files)
            if setup['case']=='press_lift':
                for row, nodes in zip(trace, data['cloth_lower'], strict=True):
                    # Native body_link_pose_w is float32. Recreate the source
                    # arrays after JSON decoding; old scorers retain that dtype
                    # in their rotation arithmetic. The geometric tolerance is
                    # unchanged and new scorers use explicit float64 internally.
                    measured = dict(row)
                    for side in ('left', 'right'):
                        measured[side+'_link_poses_xyzw'] = np.asarray(
                            row[side+'_link_poses_xyzw'], dtype=np.float32)
                    recomputed = robot_scorer.lower_clearance(measured, nodes, topology)
                    assert all(abs(recomputed[k]-row['lower_robot_clearance_m'][k])<1e-10 for k in recomputed), 'Robot clearance differs'
                    if 'upper_hardware_clearance_m' in row:
                        upper = robot_scorer.upper_hardware_clearance(row, topology)
                        assert abs(upper-row['upper_hardware_clearance_m'])<1e-10, 'Upper hardware clearance differs'
                    if 'upper_cloth_robot_clearance_m' in row:
                        upper_nodes = data['cloth_upper'][row['step']]
                        upper_cloth = robot_scorer.upper_cloth_clearance(row, upper_nodes, topology)
                        assert abs(upper_cloth-row['upper_cloth_robot_clearance_m'])<1e-10, 'Upper cloth clearance differs'
            episodes.append({'episode': path.name, 'seed': stored['seed'], 'case': stored['case'],
                             'source_hash_matches': True, 'independent_rescore_matches': True,
                             'per_step_states': len(physics), 'control_steps': len(trace), 'score': score})
            continue
        score = scorer.analyze(trace, setup['parameters'])
        if setup['case'] in ('grasp', 'frictionless') and hasattr(scorer, 'analyze_grasp'):
            score.update(scorer.analyze_grasp(trace))
        stored = json.loads((path / 'result.json').read_text())
        assert all(score[k] == stored[k] for k in score), f'Independent rescore differs: {path}'
        data = np.load(path / 'per-step.npz')
        assert all(np.isfinite(data[k]).all() for k in data.files)
        assert len(data['male_poses']) == len(data['forces']) == len(data['torques'])
        initial = json.loads((path / 'topology.json').read_text())
        reset_error = float(np.max(np.abs(data['male_poses'][0, :3] - initial['male_initial'])))
        episodes.append({'episode': path.name, 'seed': stored['seed'], 'case': stored['case'],
                         'source_hash_matches': True, 'independent_rescore_matches': True,
                         'per_step_states': len(data['forces']), 'first_step_pose_error_m': reset_error,
                         'score': score, 'current_scorer': analyze(trace, setup['parameters'])})
    return {'run': str(run), 'bundle_sha256': manifest['bundle_sha256'], 'episodes': episodes,
            'full_task_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    args.out.write_text(json.dumps(audit(args.run), indent=2) + '\n')


if __name__ == '__main__':
    main()
