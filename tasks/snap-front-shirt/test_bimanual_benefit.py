"""Reject decorative second-arm motion and invalid unsupported comparisons."""

import unittest
from analyze_bimanual_pair import benefit


class BenefitChecks(unittest.TestCase):
    def test_both_absolute_and_relative_drift_improvement_are_required(self):
        self.assertFalse(benefit(.0001,.0003,True,True,True)['qualifies_for_replication'])
        self.assertFalse(benefit(.002,.003,True,True,True)['qualifies_for_replication'])
        self.assertTrue(benefit(.0004,.001,True,True,True)['qualifies_for_replication'])

    def test_ordinary_unsupported_closure_failure_can_qualify(self):
        self.assertTrue(benefit(.001,.001,True,False,True)['qualifies_for_replication'])

    def test_invalid_unsupported_trial_cannot_qualify(self):
        self.assertFalse(benefit(.0001,.001,True,False,False)['qualifies_for_replication'])

    def test_support_must_leave_a_valid_retained_closure(self):
        self.assertFalse(benefit(.0001,.001,False,False,True)['qualifies_for_replication'])

    def test_motion_without_improvement_does_not_qualify(self):
        self.assertFalse(benefit(.001,.001,True,True,True)['qualifies_for_replication'])
