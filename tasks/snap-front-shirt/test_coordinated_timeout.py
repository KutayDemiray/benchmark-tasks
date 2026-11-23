"""Exercise response deadlines across recorded pauses and a dead guard."""

import json
from unittest import TestCase
from unittest.mock import patch
from coordinated_timeout import ActiveDeadline,paused_seconds


class DeadlineChecks(TestCase):
    def test_normal_timeout_without_guard(self):
        with patch('pathlib.Path.read_text',side_effect=FileNotFoundError), \
             patch('coordinated_timeout.time.monotonic',side_effect=[100.,105.,111.]):
            deadline=ActiveDeadline('/unused-fake-output',10.)
            self.assertEqual(deadline.remaining(),5.)
            self.assertEqual(deadline.remaining(),0.)

    def test_long_pause_preserves_remaining_work_time(self):
        initial=dict(paused_total_seconds=4.,pause_started_monotonic=None,monotonic=100.)
        resumed=dict(paused_total_seconds=304.,pause_started_monotonic=None,monotonic=405.)
        with patch('pathlib.Path.read_text',side_effect=[json.dumps(initial),json.dumps(resumed)]), \
             patch('coordinated_timeout.time.monotonic',side_effect=[100.,405.]):
            deadline=ActiveDeadline('/unused-fake-output',10.)
            self.assertEqual(deadline.remaining(),5.)

    def test_dead_guard_does_not_extend_deadline_forever(self):
        stopped=dict(paused_total_seconds=2.,pause_started_monotonic=10.,monotonic=20.)
        with patch('pathlib.Path.read_text',return_value=json.dumps(stopped)):
            self.assertEqual(paused_seconds('/unused-fake-output',15.),7.)
            self.assertEqual(paused_seconds('/unused-fake-output',25.),17.)
            self.assertEqual(paused_seconds('/unused-fake-output',10000.),22.)
