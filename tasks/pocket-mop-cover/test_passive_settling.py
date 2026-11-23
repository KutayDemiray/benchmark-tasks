"""Reject retrospective, incomplete, or interrupted passive settling holds."""

import unittest
from passive_settling import PassiveHold


class PassiveSettlingTests(unittest.TestCase):
    def test_every_native_step_must_pass_both_speed_checks(self):
        for bad in ((.031, 0., 0., 0., .005), (0., .031, 0., 0., .005),
                    (0., 0., 4., 0., .005), (0., 0., 0., 4., .005),
                    (0., 0., 0., 0., .011)):
            with self.subTest(bad=bad):
                hold=PassiveHold(480)
                for _ in range(959):
                    self.assertFalse(hold.observe(0., 0., 0., 0., .005))
                self.assertFalse(hold.observe(*bad))
                self.assertEqual(hold.contiguous_steps, 0)
                for _ in range(959):
                    self.assertFalse(hold.observe(0., 0., 0., 0., .005))
                self.assertTrue(hold.observe(0., 0., 0., 0., .005))

    def test_native_hold_finishes_only_at_control_boundary(self):
        hold=PassiveHold(480)
        hold.observe(.1, 0., 0., 0., .005)
        for _ in range(960):
            self.assertFalse(hold.observe(.02, .02, 0., 0., .005))
        for _ in range(22):
            self.assertFalse(hold.observe(.02, .02, 0., 0., .005))
        self.assertTrue(hold.observe(.02, .02, 0., 0., .005))
        self.assertEqual(hold.first_stop_step, 984)

    def test_nonfinite_readback_cannot_qualify(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(AssertionError):
                PassiveHold(480).observe(value, 0., 0., 0., .005)
