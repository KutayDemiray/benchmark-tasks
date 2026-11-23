"""Exercise the observation boundary and installed Dream client with no remote session."""

import unittest
from unittest.mock import patch
import numpy as np

from policy_contract import CAMERAS,embodiment_info,observation,validate_action


class FakeSession:
    def __init__(self, fail=False):
        self.calls=[]
        self.fail=fail
    def check(self, episode):self.calls.append('check')
    def start(self, episode):self.calls.append('start')
    def observe(self, frames, state, prompt, timestamp):
        self.calls.append(('observe',timestamp))
        assert len(frames)==3 and state.shape==(14,)
    def act(self, commanded):
        self.calls.append('act')
        if self.fail:raise RuntimeError('injected offline failure')
        from inspect_robots_dream.session import Chunk
        return Chunk(rows=np.zeros((48,14)),rtt_s=.01,attempts=1)
    def close(self):self.calls.append('close')


class ContractChecks(unittest.TestCase):
    def sample(self, timestamp=0.):
        return observation({name:np.zeros((360,640,3),dtype=np.uint8) for name in CAMERAS},np.zeros(14),timestamp)

    def test_policy_observation_contains_only_public_fields(self):
        sample=self.sample(.05)
        self.assertEqual(set(sample.state),{'joint_pos'})
        self.assertFalse(sample.extra)
        self.assertEqual(sample.state_time,.05)
        self.assertTrue(all(image.shape==(176,320,3) for image in sample.images.values()))
        self.assertTrue(all(t==.05 for t in sample.image_times.values()))

    def test_bad_actions_are_rejected_and_valid_targets_not_rewritten(self):
        for invalid in (np.zeros(13),np.zeros((2,7)),np.full(14,np.nan),np.full(14,np.inf)):
            with self.assertRaises(ValueError):validate_action(invalid)
        np.testing.assert_array_equal(validate_action(np.arange(14)),np.arange(14))

    def test_fake_session_uses_installed_policy_without_network(self):
        from inspect_robots import Scene
        from inspect_robots_dream.policy import DreamPolicy
        session=FakeSession()
        with patch('socket.socket',side_effect=AssertionError('Network access forbidden in offline test')):
            policy=DreamPolicy(session=session,preroll_steps=0)
            policy.bind(embodiment_info())
            policy.reset(Scene(id='snap-shirt-offline-test',instruction='Fasten the upper snaps.'))
            chunk=policy.act(self.sample())
            self.assertEqual(len(chunk.actions),1)
            self.assertEqual(chunk.actions[0].data.shape,(14,))
            policy.close()
        self.assertIn('act',session.calls)

    def test_fake_session_errors_remain_errors(self):
        from inspect_robots import Scene
        from inspect_robots_dream.policy import DreamPolicy
        policy=DreamPolicy(session=FakeSession(fail=True),preroll_steps=0)
        with patch('socket.socket',side_effect=AssertionError('Network access forbidden in offline test')):
            policy.bind(embodiment_info())
            policy.reset(Scene(id='snap-shirt-offline-test',instruction='Fasten the upper snaps.'))
            with self.assertRaisesRegex(RuntimeError,'injected offline failure'):
                policy.act(self.sample())
            policy.close()
