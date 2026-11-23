"""Coverage and evidence checks for parallel material-audit aggregation."""

import unittest

from audit_cover_parallel import merge_parts


def part(start,stop):
    hashes={f'physics-chunks/{i:05d}.npz':str(i) for i in range(start,stop)}
    hashes['geometry.npz']='geometry'
    return dict(chunk_range=[start,stop],checked_chunks=stop-start,physics_step_offset=start*480,
        physics_steps=(stop-start)*480,source_sha256=hashes,complete=False,finite=True,
        max_abs_edge_strain=.1,self_crossing_penetration_upper_bound_m=.0003,
        max_seam_gap_m=0.,no_attachment=True,violations=[],native_final_two_second_speed_m_s=.02,
        difference_final_two_second_speed_m_s=.01,numerical_checks_passed=False)


class ParallelAuditTests(unittest.TestCase):
    def test_completed_diagnostic_keeps_its_exclusion_after_merging(self):
        first,last=part(0,3),part(3,6)
        for value in (first,last):
            value.update(recording_status='completed',diagnostic_only=True,benchmark_eligible=False)
        result=merge_parts([first,last],[(0,3),(3,6)],2880,120,480)
        self.assertTrue(result['complete'] and result['recorded_numerical_checks_passed'])
        self.assertTrue(result['diagnostic_only'])
        self.assertFalse(result['benchmark_eligible'])

    def test_stopped_controller_cannot_receive_a_positive_certificate(self):
        first,last=part(0,3),part(3,6)
        for value in (first,last):
            value['recording_status']='error'
        result=merge_parts([first,last],[(0,3),(3,6)],2880,120,480)
        self.assertTrue(result['complete_recorded_step_coverage'])
        self.assertTrue(result['recorded_numerical_checks_passed'])
        self.assertTrue(result['diagnostic_only'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['numerical_checks_passed'])

    def test_complete_coverage_merges_bounds_and_retains_final_window(self):
        first,last=part(0,3),part(3,6)
        first['max_abs_edge_strain']=.19
        first['native_final_two_second_speed_m_s']=4.
        result=merge_parts([first,last],[(0,3),(3,6)],2880,120,480)
        self.assertTrue(result['complete'] and result['numerical_checks_passed'])
        self.assertEqual(result['max_abs_edge_strain'],.19)
        self.assertEqual(result['native_final_two_second_speed_m_s'],.02)

    def test_any_worker_violation_rejects_whole_run(self):
        for key,value in [('finite',False),('no_attachment',False),('max_abs_edge_strain',.201),
                          ('violations',[dict(physics_step=27,colliders=['test'])])]:
            first,last=part(0,3),part(3,6)
            first[key]=value
            result=merge_parts([first,last],[(0,3),(3,6)],2880,120,480)
            self.assertFalse(result['numerical_checks_passed'])

    def test_gap_duplicate_missing_file_or_changed_source_is_rejected(self):
        for mutation in ('offset','count','file','hash'):
            first,last=part(0,3),part(3,6)
            if mutation=='offset':last['physics_step_offset']=0
            if mutation=='count':last['physics_steps']-=24
            if mutation=='file':last['source_sha256'].pop('physics-chunks/00004.npz')
            if mutation=='hash':last['source_sha256']['geometry.npz']='changed'
            with self.assertRaises(AssertionError):
                merge_parts([first,last],[(0,3),(3,6)],2880,120,480)


if __name__=='__main__':
    unittest.main()
