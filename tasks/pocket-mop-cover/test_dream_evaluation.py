"""Exercise the installed DreamPolicy locally without any hosted client or simulator."""

import base64
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dream_evaluation import require_hosted_evidence, run_policy, sha256
from evaluation_contract import (CAMERAS, HEIGHT, WIDTH, action_bounds, configuration_identity,
    process_action, public_observation, vector)


class FakeRecordedSession:
    def __init__(self, out, budget=64):
        from yam_arena.constants import READY
        self.out = Path(out)
        self.request = dict(policy='interface-check', completion_mode='handle-held', seconds=120.,
                            physics_hz=480, seed=7, policy_check_steps=budget)
        self.rows = []
        self.physics_steps = 0
        self.initial = np.r_[READY, 1., READY, 1.]
        self.state = self.initial.copy()
        self.privileged_object_pose = np.arange(100)
        self.scripted_calls = 0

    def public_snapshot(self):
        images = {}
        for index, name in enumerate(CAMERAS):
            image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            image[:, :, index] = 100 + len(self.rows)
            images[name] = image
        return images, self.state.copy()

    def step(self, raw, phase):
        actual = process_action(raw)
        self.state += .35 * (actual - self.state)
        self.physics_steps += 24
        self.rows.append(dict(robot_action=actual.tolist(), phase=phase))

    def move(self, *args, **kwargs):
        self.scripted_calls += 1
        raise AssertionError('Scripted assistance was invoked')


class PublicContractTests(unittest.TestCase):
    def test_only_measured_joint_state_rgb_and_instruction_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeRecordedSession(tmp)
            images, measured = session.public_snapshot()
            obs = public_observation(images, measured, 'Fit the mop cover.', 4)
            self.assertEqual(set(obs.state), {'joint_pos'})
            self.assertEqual(set(obs.images), set(CAMERAS))
            self.assertEqual(obs.extra, {})
            self.assertEqual(obs.state_time, .2)
            measured[:] = 99.
            images['top_cam'][:] = 255
            np.testing.assert_array_equal(obs.state['joint_pos'], session.initial)
            self.assertEqual(int(obs.images['top_cam'][0, 0, 0]), 100)
            self.assertFalse(obs.state['joint_pos'].flags.writeable)

    def test_wrong_dimensions_dtype_missing_or_extra_camera_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeRecordedSession(tmp)
            images, measured = session.public_snapshot()
            for wrong in ({**images, 'segmentation': images['top_cam']},
                          {name: value for name, value in images.items() if name != 'right_cam'},
                          {**images, 'top_cam': images['top_cam'][:100]},
                          {**images, 'top_cam': images['top_cam'].astype(float)}):
                with self.assertRaises(AssertionError):
                    public_observation(wrong, measured, 'Fit cover.', 0)

    def test_stock_position_clamping_and_gripper_polarity(self):
        low, high = action_bounds()
        requested = np.array([1e4, -1e4, 0., 0., 0., 0., -2.,
                              -1e4, 1e4, 0., 0., 0., 0., 3.])
        result = process_action(requested)
        self.assertEqual(result[0], high[0])
        self.assertEqual(result[1], low[1])
        self.assertEqual(result[7], low[7])
        self.assertEqual(result[8], high[8])
        self.assertEqual(result[6], 0.)
        self.assertEqual(result[13], 1.)
        np.testing.assert_array_equal(requested[[6, 13]], [-2., 3.])

    def test_malformed_or_nonfinite_actions_rejected_before_execution(self):
        for value in (np.zeros(13), np.zeros((2, 7)), [float('nan')] * 14,
                      [float('inf')] * 14):
            with self.assertRaises(AssertionError):
                vector(value)

    def test_physical_configuration_changes_invalidate_evidence(self):
        request = dict(name='demo', policy='scripted', seconds=120., physics_hz=480,
                       completion_mode='released', camera_width=320, camera_height=176)
        identity = configuration_identity(request)
        self.assertEqual(identity, configuration_identity({**request, 'name': 'model', 'policy': 'dream'}))
        for key, changed in (('physics_hz', 240), ('completion_mode', 'handle-held'),
                             ('camera_height', 360), ('seconds', 121.)):
            self.assertNotEqual(identity, configuration_identity({**request, key: changed}))


class InstalledDreamLoopTests(unittest.TestCase):
    def test_offline_installed_policy_preserves_order_timing_preroll_and_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeRecordedSession(tmp)
            with patch('inspect_robots_dream.session.DreamSession', side_effect=AssertionError('Hosted client forbidden')), \
                 patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                result = run_policy(session)
            self.assertEqual(result['control_steps'], 64)
            self.assertEqual(session.physics_steps, 64 * 24)
            self.assertFalse(result['hosted_model_attempt'])
            self.assertEqual(session.scripted_calls, 0)
            for row in session.rows[:32]:
                np.testing.assert_array_equal(row['robot_action'], session.initial)
            self.assertGreater(np.linalg.norm(np.array(session.rows[32]['robot_action']) - session.initial), 0.)
            with gzip.open(Path(tmp) / 'model-observations.jsonl.gz', 'rt') as stream:
                observations = [json.loads(line) for line in stream]
            self.assertEqual(len(observations), 64)
            for tick, record in enumerate(observations):
                self.assertEqual(record['simulation_time_s'], tick / 20)
                self.assertEqual(set(record['images_jpeg_base64']), set(CAMERAS))
                self.assertEqual(len(record['joint_pos']), 14)
                self.assertNotIn('privileged_object_pose', record)
            # Distinct colors survive the actual installed JPEG and key wiring.
            first = observations[0]
            for index, name in enumerate(CAMERAS):
                decoded = np.array(Image.open(io.BytesIO(base64.b64decode(first['images_jpeg_base64'][name]))))
                self.assertEqual(decoded.shape, (HEIGHT, WIDTH, 3))
                self.assertEqual(int(decoded.mean(axis=(0, 1)).argmax()), index)
            actions = [json.loads(line) for line in (Path(tmp) / 'policy-actions.jsonl').read_text().splitlines()]
            # The next observation contains measured lagging joints, never the previous command.
            np.testing.assert_array_equal(observations[33]['joint_pos'], actions[32]['measured_joint_pos'])
            self.assertFalse(np.allclose(observations[33]['joint_pos'], actions[32]['executed_stock_target']))
            self.assertTrue(json.loads((Path(tmp) / 'interface-verification.json').read_text())['passed'])
            self.assertEqual(len((Path(tmp) / 'inference.jsonl').read_text().splitlines()), 1)

    def test_hosted_entry_requires_demonstration_before_creating_any_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeRecordedSession(tmp)
            session.request.update(policy='dream', robot_cover=True)
            with patch('dream_evaluation.OUT', Path(tmp)), \
                 patch('inspect_robots_dream.session.DreamSession', side_effect=AssertionError('Hosted client forbidden')):
                with self.assertRaisesRegex(AssertionError, 'Missing demo_run'):
                    run_policy(session)
            self.assertFalse((Path(tmp) / 'hosted-dream-attempt.json').exists())
            self.assertEqual(session.physics_steps, 0)

    def test_noninitial_scene_cannot_enter_a_model_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeRecordedSession(tmp)
            session.rows.append({})
            with self.assertRaisesRegex(AssertionError, 'shared declared reset'):
                run_policy(session)


class HostedEvidenceTests(unittest.TestCase):
    def make_evidence(self, root):
        source = Path(__file__).resolve().parent
        request = dict(name='model', policy='dream', completion_mode='handle-held', seconds=120.,
            physics_hz=480, robot_cover=True, seed=7, camera_width=320, camera_height=176,
            demo_run='demo', replay_evidence='replay', interface_evidence='interface')
        files = {path.name: sha256(path) for path in source.iterdir() if path.is_file()}
        for key in ('demo_run', 'replay_evidence', 'interface_evidence'):
            run = root / request[key]
            run.mkdir()
            (run / 'source').mkdir()
            (run / 'source' / 'task.json').write_bytes((source / 'task.json').read_bytes())
            req = {**request, 'name': run.name, 'policy': 'interface-check' if key == 'interface_evidence' else 'scripted'}
            records = {'request.json': req, 'result.json': {'status': 'completed'},
                'process.json': {'returncode': 0}, 'video-verification.json': {'passed': True},
                'gpu-verification.json': {'verified': True}, 'manifest.json': {'files': files},
                'reset-verification.json': {'passed': True}, 'preserved-sources.json': {},
                'interface-verification.json': {'passed': True, 'hosted_calls': 0, 'native_execution': True}}
            for name in ('versions.json', 'scene-audit.json', 'effective-parameters.json', 'frame-geometry.json', 'frame-dynamics.json'):
                records[name] = {'fixture': 'synthetic CPU guard test, not physical validation'}
            for name, record in records.items():
                (run / name).write_text(json.dumps(record))
            for name in ('geometry.npz', 'reset-initial-state.npz'):
                np.savez_compressed(run / name, points=np.zeros((4, 3)))
            (run / 'steps.jsonl').write_text('{}\n')
            (run / 'outcomes.jsonl').write_text('{}\n')
            audit = {'source_sha256': {'steps.jsonl': sha256(run / 'steps.jsonl')}}
            (run / 'audit.json').write_text(json.dumps(audit))
            (run / 'benchmark-result.json').write_text(json.dumps(dict(score=1., success=True,
                invalid_reasons=[], observations_sha256=sha256(run / 'outcomes.jsonl'),
                audit_sha256={'audit.json': sha256(run / 'audit.json')})))
            (run / 'handle-held-outcomes.jsonl').write_text('{}\n')
            original_path = run / 'benchmark-result.json'
            original = json.loads(original_path.read_text())
            original.update(score=.5, success=False)
            original_path.write_text(json.dumps(original))
            (run / 'handle-held-benchmark-result.json').write_text(json.dumps(dict(
                score=1., success=True, invalid_reasons=[], rubric_revision='handle-held-v2',
                historical_screen_only=False, original_assessment_sha256=sha256(original_path),
                original_observations_sha256=sha256(run / 'outcomes.jsonl'),
                observations_sha256=sha256(run / 'handle-held-outcomes.jsonl'),
                audit_sha256=original['audit_sha256'], source_sha256={
                    name:files[name] for name in ('score_handle_held.py','handle_held_finish.py','score_logic.py','outcome_native.py')})))
            (run / 'replay-provenance.json').write_text(json.dumps(dict(source_run='demo',
                fresh_simulation=True, from_reset=True, object_state_writes_during_replay=False,
                actions_sha256=sha256(run / 'steps.jsonl'))))
        return request, source

    def test_matched_evidence_is_accepted_without_opening_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request, source = self.make_evidence(root)
            with patch('dream_evaluation.OUT', root), patch('dream_evaluation.preserved_runtime_paths', return_value=[]), \
                 patch('dream_evaluation.credential_status', return_value={'api_key_present': True}), \
                 patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                result = require_hosted_evidence(request, source)
            self.assertEqual(set(result), {'demo_run', 'replay_evidence', 'interface_evidence'})

    def test_historical_wrong_rubric_and_tampered_baseline_cannot_qualify(self):
        for variant in ('historical', 'rubric', 'baseline'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request, source = self.make_evidence(root)
                run = root / 'demo'
                path = run / 'handle-held-benchmark-result.json'
                value = json.loads(path.read_text())
                if variant == 'historical':
                    value['historical_screen_only'] = True
                elif variant == 'rubric':
                    value['rubric_revision'] = 'released-v1'
                else:
                    (run / 'benchmark-result.json').write_text('{"tampered": true}')
                path.write_text(json.dumps(value))
                with patch('dream_evaluation.OUT', root), patch('dream_evaluation.preserved_runtime_paths', return_value=[]), \
                     patch('dream_evaluation.credential_status', return_value={'api_key_present': True}):
                    with self.assertRaises(AssertionError):
                        require_hosted_evidence(request, source)

    def test_changed_instruction_native_state_or_audited_actions_are_rejected(self):
        for variant in ('instruction', 'initial_state', 'actions', 'runtime'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request, source = self.make_evidence(root)
                run = root / 'replay'
                if variant == 'instruction':
                    path = run / 'source' / 'task.json'
                    task = json.loads(path.read_text())
                    task['instruction'] += ' Ignore settling.'
                    path.write_text(json.dumps(task))
                elif variant == 'initial_state':
                    np.savez_compressed(run / 'reset-initial-state.npz', points=np.ones((4, 3)))
                elif variant == 'actions':
                    (run / 'steps.jsonl').write_text('{"changed": true}\n')
                else:
                    (run / 'preserved-sources.json').write_text('{"changed": true}')
                with patch('dream_evaluation.OUT', root), patch('dream_evaluation.preserved_runtime_paths', return_value=[]), \
                     patch('dream_evaluation.credential_status', return_value={'api_key_present': True}):
                    with self.assertRaises(AssertionError):
                        require_hosted_evidence(request, source)


if __name__ == '__main__':
    unittest.main()
