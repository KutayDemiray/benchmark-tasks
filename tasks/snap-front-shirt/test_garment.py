"""Topology, penetration and intermediate-credit guards for the garment stage."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from garment_contact_audit import bar_contacts,segment_triangle_distances,initial_hook_rail_gap,hook_rail_penetration
from garment_mesh import pattern,topology_audit
from garment_runner import passive_score,verify_world_joint
from hanger_shapes import triangle_box_intersections,configure_wooden_hanger,wooden_beams


class GarmentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((Path(__file__).parent/'parameters.json').read_text())
        cls.data=pattern(cls.cfg)

    def test_sewn_pattern_has_only_front_hem_and_two_cuff_openings(self):
        d=self.data
        self.assertEqual(d['audit']['connected_components'],1)
        self.assertEqual(d['audit']['boundary_loops'],3)
        self.assertTrue(d['audit']['oriented'])
        self.assertEqual(d['audit']['referenced_vertices'],len(d['initial']))
        self.assertEqual(len(d['rivets']),14)
        self.assertEqual(self.cfg['completion']['front_snap_count'],1)
        self.assertEqual(self.cfg['completion']['target_snap_index'],0)

    def test_duplicate_face_is_nonmanifold(self):
        d=self.data
        audit=topology_audit(d['initial'],np.vstack((d['faces'],d['faces'][200])))
        self.assertEqual(audit['max_edge_valence'],3)

    def test_bar_piercing_triangle_is_detected_even_with_clear_vertices(self):
        nodes=np.array([[-.012,0,-.092],[.012,0,-.092],[0,0,-.068]])
        result=bar_contacts(nodes,np.array([[0,1,2]]),[0,0,0,0,0,0,1],self.cfg['hanger'])
        self.assertEqual(result[2]['crossed_triangles'],[0])
        self.assertEqual(result[2]['maximum_vertex_penetration_m'],0.)

    def test_clear_triangle_is_not_penetration(self):
        nodes=np.array([[-.012,0,-.02],[.012,0,-.02],[0,.01,-.01]])
        result=bar_contacts(nodes,np.array([[0,1,2]]),[0,0,0,0,0,0,1],self.cfg['hanger'])
        self.assertTrue(all(not b['crossed_triangles'] for b in result))

    def test_hook_rail_overlap_witness_rejects_archived_settled_pose(self):
        h=self.cfg['hanger']
        self.assertGreater(initial_hook_rail_gap(h),0.)
        initial=[*h['origin_m'],0,0,0,1]
        self.assertEqual(hook_rail_penetration(initial,h)['penetration_lower_bound_m'],0.)
        penetrated=initial.copy()
        penetrated[2]=1.241049
        self.assertGreater(hook_rail_penetration(penetrated,h)['penetration_lower_bound_m'],.007)

    def test_triangle_box_detects_face_crossing_with_all_vertices_outside(self):
        triangles=np.array([[[-2,0,0],[2,0,0],[0,2,0]],
                            [[-2,0,2],[2,0,2],[0,2,2]]],dtype=float)
        self.assertEqual(triangle_box_intersections(triangles,[.5,.5,.5]).tolist(),[True,False])

    def test_wooden_hanger_width_includes_its_physical_cross_section(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['garment_test']={}
        configure_wooden_hanger(cfg)
        h=cfg['hanger']
        for b in wooden_beams(h):
            half_extents=np.abs(b['rotation'])@(b['size']/2)
            self.assertAlmostEqual(abs(b['center'][1])+half_extents[1],h['width_m']/2)
            self.assertAlmostEqual(2*half_extents[0],.014)
            self.assertTrue(np.allclose(b['rotation'].T@b['rotation'],np.eye(3)))

    def test_segment_face_distance_covers_interior_of_large_triangle(self):
        tri=np.array([[[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]]])
        value=segment_triangle_distances(tri,[0.,0.,1.],[0.,0.,2.])
        self.assertAlmostEqual(value[0],1.)

    def test_segment_edge_distance_covers_parallel_endpoints_outside_face(self):
        tri=np.array([[[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]]])
        value=segment_triangle_distances(tri,[-2.,0.,.5],[2.,0.,.5])
        self.assertAlmostEqual(value[0],.5)

    def rows(self):
        hardware={'hanger':[*self.cfg['hanger']['origin_m'],0,0,0,1]}
        for i in range(7):
            hardware[f'socket{i}']=[.28,0,1-i*.08,0,0,0,1]
            hardware[f'male{i}']=[.28,0,1.014-i*.08,0,0,0,1]
        return [dict(step=i,physics=dict(hardware=copy.deepcopy(hardware),
                    mean_cloth_position_m=[.28,0,.85],min_cloth_z_m=.45)) for i in range(50)]

    def test_passive_success_never_awards_full_task_credit(self):
        result=passive_score(self.rows(),self.cfg,'diagnostic_completed',.02,.0001,True)
        self.assertTrue(result['diagnostic_pass'])
        self.assertEqual(result['full_task_score'],0.)
        self.assertFalse(result['full_task_success'])

    def test_observing_past_strain_failure_cannot_turn_it_into_success(self):
        result=passive_score(self.rows(),self.cfg,'invalid_fabric_state',.25,.0001,True)
        self.assertFalse(result['diagnostic_pass'])
        self.assertFalse(result['checks']['intact'])

    def test_uncommanded_seating_fails_passive_gate(self):
        rows=self.rows()
        rows[20]['physics']['hardware']['male0'][2]=1.004
        result=passive_score(rows,self.cfg,'diagnostic_completed',.02,.0001,True)
        self.assertFalse(result['diagnostic_pass'])
        self.assertFalse(result['checks']['no_self_closing'])

    def test_retained_snap_at_6p8mm_is_detected(self):
        rows=self.rows()
        rows[20]['physics']['hardware']['male0'][2]=1.0068
        result=passive_score(rows,self.cfg,'diagnostic_completed',.02,.0001,True)
        self.assertFalse(result['checks']['no_self_closing'])
        self.assertEqual(result['self_seated_events'][0]['pair'],0)

    def test_loaded_loop_cannot_pass_after_dropping_its_payload(self):
        rows=self.rows()
        cfg=copy.deepcopy(self.cfg)
        cfg['garment_test']={'case':'loaded_sleeve_probe'}
        cfg['garment']['snap_heights_from_hem_m']=[]
        for row in rows:
            row['physics']['hardware']['payload']=[.28,0,.02,0,0,0,1]
        result=passive_score(rows,cfg,'diagnostic_completed',.02,0.,True)
        self.assertFalse(result['diagnostic_pass'])
        self.assertFalse(result['checks']['payload_supported'])
        self.assertEqual(result['full_task_score'],0.)

    def test_only_declared_diagnostic_world_mount_is_allowed(self):
        meta=dict(hardware={'hanger':'/task/hanger'},diagnostic_world_mounts=True,
                  allowed_diagnostic_world_joint='/task/hanger/diagnostic_mount')
        self.assertTrue(verify_world_joint('/task/hanger/diagnostic_mount',[],['/task/hanger'],meta))
        with self.assertRaises(AssertionError):
            verify_world_joint('/task/hidden_mount',[],['/task/hanger'],meta)
        with self.assertRaises(AssertionError):
            verify_world_joint('/task/hanger/diagnostic_mount',[],['/task/shirt'],meta)
        meta['diagnostic_world_mounts']=False
        with self.assertRaises(AssertionError):
            verify_world_joint('/task/hanger/diagnostic_mount',[],['/task/hanger'],meta)

    def test_fixed_support_cannot_validate_free_hanger_or_full_task(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['garment_test']={'fixed_support':True}
        result=passive_score(self.rows(),cfg,'diagnostic_completed',.02,0.,True)
        self.assertTrue(result['diagnostic_pass'])
        self.assertFalse(result['free_hanger_support_validated'])
        self.assertFalse(result['full_task_success'])
        self.assertEqual(result['full_task_score'],0.)


if __name__=='__main__':
    unittest.main()
