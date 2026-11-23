"""The generic audit keeps native collision checks and needs no scripted phases."""

import ast
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_episode_contacts import HELPERS, audit

HERE = Path(__file__).resolve().parent


class EpisodeContactTests(unittest.TestCase):
    def make_run(self, run, collision=False, count=2):
        def save(name, value):
            (run / name).write_text(json.dumps(value))

        (run / 'source').mkdir()
        hashes = {}
        for name in HELPERS:
            content = (HERE / name).read_bytes()
            (run / 'source' / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        save('manifest.json', {'files': hashes})
        save('request.json', dict(physics_hz=20, robot_cover=True, robot_reopen=True,
            table_origin=False, policy='dream', completion_mode='handle-held'))
        save('result.json', dict(status='completed', control_steps=2, physics_steps=2))
        save('frame-names.json', dict(body_names=['Bridge'], joint_names=['HingeLeft', 'HingeRight']))
        cube = (np.array(list(itertools.product((-1., 1.), repeat=3))) * .02).tolist()
        shape = dict(name='Box', body='Bridge', points=cube, enabled=True)
        save('frame-geometry.json', dict(colliders=[shape]))
        robots = {side: dict(body_names=['base'], indices=[0],
            colliders=[dict(name='Box', body='base', points=cube)]) for side in ('left', 'right')}
        save('robot-colliders.json', robots)
        (run / 'scene.usda').write_text('Synthetic geometric audit fixture; no simulation.')
        save('versions.json', {})
        robots['_provenance'] = dict(source_run=run.name, scene_modified=False,
            source_sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                for name in ('scene.usda', 'robot-colliders.json', 'versions.json', 'manifest.json')})
        save('robot-cooked-colliders.json', robots)
        save('scene-audit.json', dict(table_center_m=[0., 0., -1.], table_size_m=[10., 10., 1.]))
        (run / 'steps.jsonl').write_text('\n'.join(json.dumps(dict(step=i, phase='dream')) for i in range(2)))
        (run / 'physics-chunks').mkdir()
        pose = lambda position: np.tile([*position, 0., 0., 0., 1.], (count, 1, 1))
        np.savez(run / 'physics-chunks/00000.npz', frame_body_poses=pose([2., 2., 1.]),
            frame_joint_positions=np.zeros((count, 2)), left_body_poses=pose([0., 0., 2.]),
            right_body_poses=pose([2., 2., 1.] if collision else [-2., 0., 2.]))

    def test_contact_algorithm_is_identical_to_original(self):
        old = ast.parse((HERE / 'audit_frame_robot.py').read_text())
        new = ast.parse((HERE / 'audit_episode_contacts.py').read_text())
        original = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
        generic = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == '_audit')
        boundary = next(i for i, n in enumerate(original.body)
            if isinstance(n, ast.If) and 'release_probe_run' in ast.unparse(n.test))
        prefix = original.body[:boundary]
        dumps = lambda nodes: [ast.dump(n, include_attributes=False) for n in nodes]
        self.assertTrue(any(dumps(generic.body[i:i + len(prefix)]) == dumps(prefix)
                            for i in range(len(generic.body))))

    def test_model_phases_need_no_release_or_retention_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_run(run)
            result = audit(run)
            self.assertTrue(result['contact_checks_passed'])
            self.assertTrue(result['robot_table_and_self_contacts_audited'])
            self.assertEqual(result['physics_steps'], 2)
            self.assertFalse(result['full_task_success'])
            self.assertFalse(result['frame_handling_passed'])
            self.assertIn('assessment-source/audit_episode_contacts.py', result['source_sha256'])
            for name, expected in result['source_sha256'].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), expected)
            rows = [dict(step=i, phase='arbitrary-label') for i in range(2)]
            (run / 'steps.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            again = audit(run)
            for key in ('contact_checks_passed', 'max_convex_penetration_m', 'max_gear_error_deg'):
                self.assertEqual(again[key], result[key])

    def test_real_geometric_penetration_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_run(run, collision=True)
            result = audit(run)
            self.assertFalse(result['contact_checks_passed'])
            self.assertGreater(result['max_convex_penetration_m'], .001)
            self.assertTrue(any(pair['violating_steps'] for pair in result['overlapping_pairs']))

    def test_missing_native_states_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_run(run, count=1)
            with self.assertRaises(AssertionError):
                audit(run)

    def test_changed_archived_helper_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_run(run)
            with (run / 'source/audit_mechanism.py').open('a') as stream:
                stream.write('\n# changed\n')
            with self.assertRaises(AssertionError):
                audit(run)


if __name__ == '__main__':
    unittest.main()
