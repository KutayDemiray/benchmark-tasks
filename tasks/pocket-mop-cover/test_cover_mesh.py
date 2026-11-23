"""Two-pocket topology and physical enclosure; does not qualify simulation."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from cover_mesh import cover_mesh
from mesh import audit
from measurements import inside,pocket_seating
from robot_gate import fabric


class CoverGeometryTests(unittest.TestCase):
    def setUp(self):
        config=json.loads(Path(__file__).with_name('parameters.json').read_text())
        self.points,self.faces,self.meta=cover_mesh(fabric(config))

    def test_continuous_cover_has_two_distinct_enclosures(self):
        self.assertTrue(audit(self.points,self.faces)['connected'])
        queries=[[.04,0,.008],[.40,0,.008],[.22,0,.008],[.04,0,.03]]
        for pocket,expected in zip(self.meta['pockets'],[[True,False,False,False],[False,True,False,False]]):
            self.assertEqual(inside(queries,self.points,np.asarray(pocket['closed_faces'])).tolist(),expected)
        self.assertEqual(len(self.meta['middle_join_node_ids']),29)
        self.assertAlmostEqual(self.points[:,0].max()-self.points[:,0].min(),.44)

    def test_both_ends_inside_and_on_top_counterexample(self):
        # A 400 mm head centered at .22 has leading tips at .02 and .42.
        for i,(center,rotation) in enumerate([([.07,0,.006],np.eye(3)),
                ([.37,0,.006],Rotation.from_euler('z',180,degrees=True).as_matrix())]):
            pocket=self.meta['pockets'][i]
            values=dict(markers=pocket['markers'],faces=np.asarray(pocket['faces']),profile='beveled',rotation=rotation)
            valid=pocket_seating(self.points,np.asarray(pocket['closed_faces']),np.asarray(center),[.1,.1,.006],**values)
            top=pocket_seating(self.points,np.asarray(pocket['closed_faces']),np.asarray(center)+[0,0,.04],[.1,.1,.006],**values)
            self.assertTrue(valid['seated_samples'])
            self.assertFalse(top['seated_samples'])

    def test_global_transform_preserves_pocket_volumes(self):
        rotation=Rotation.from_euler('xyz',[12,23,34],degrees=True).as_matrix();translation=np.array([.3,-.4,.8])
        points=self.points@rotation.T+translation
        queries=np.array([[.04,0,.008],[.40,0,.008],[.22,0,.008]])
        for pocket in self.meta['pockets']:
            closed=np.asarray(pocket['closed_faces'])
            np.testing.assert_array_equal(inside(queries,self.points,closed),inside(queries@rotation.T+translation,points,closed))

    def test_seating_orientation_follows_the_frame_during_lift(self):
        # A correctly fitted cover remains correctly oriented when the whole
        # assembly tilts; flipping only the textile must still fail.
        common=Rotation.from_euler('xyz',[80,23,34],degrees=True).as_matrix()
        translation=np.array([.3,-.4,.8])
        for i,(center,end_rotation) in enumerate([([.07,0,.006],np.eye(3)),
                ([.37,0,.006],Rotation.from_euler('z',180,degrees=True).as_matrix())]):
            pocket=self.meta['pockets'][i]
            values=dict(markers=pocket['markers'],faces=np.asarray(pocket['faces']),profile='beveled')
            moved=pocket_seating(self.points@common.T+translation,np.asarray(pocket['closed_faces']),
                common@center+translation,[.1,.1,.006],rotation=common@end_rotation,**values)
            flipped=self.points.copy();flipped[:,1:]*=-1
            wrong=pocket_seating(flipped@common.T+translation,np.asarray(pocket['closed_faces']),
                common@center+translation,[.1,.1,.006],rotation=common@end_rotation,**values)
            self.assertTrue(moved['seated_samples'])
            self.assertTrue(moved['orientation_correct'])
            self.assertFalse(wrong['orientation_correct'])
