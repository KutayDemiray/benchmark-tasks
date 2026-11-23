"""Keep the unvalidated policy runtime behind the complete passive gate."""

import copy
import unittest
from evaluation_runner import selected_configuration


class EvaluationGateChecks(unittest.TestCase):
    def report(self):
        configuration=dict(name='fresh-process',iterations=64,pitch=.01)
        episodes=[]
        for seed,seconds in ((100,10),(101,10),(100,300)):
            episodes.append(dict(seed=seed,seconds=seconds,passed=True,worker={'returncode':0},
                result=dict(diagnostic_pass=True,invalid_reasons=[],control_steps=seconds*10,
                            physics_steps_observed=seconds*960+1)))
        return dict(gate_passed=True,selected_configuration=configuration,
                    candidates=[dict(configuration=configuration,passed=True,episodes=episodes)])

    def test_only_complete_passive_evidence_selects_a_configuration(self):
        self.assertEqual(selected_configuration(self.report())['iterations'],64)
        for change in ('pending','missing_horizon','invalid','truncated'):
            report=self.report()
            if change=='pending':report['gate_passed']=False
            elif change=='missing_horizon':report['candidates'][0]['episodes'].pop()
            elif change=='invalid':report['candidates'][0]['episodes'][-1]['result']['invalid_reasons']=['cloth_strain_limit']
            else:report['candidates'][0]['episodes'][-1]['result']['control_steps']=2999
            with self.assertRaises(AssertionError):selected_configuration(report)
