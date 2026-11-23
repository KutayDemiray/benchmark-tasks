"""Independent positive/negative cases for contact audit geometry."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from self_contact import crossings, separating_translations
from audit_mechanism import features,penetration


class ContactAuditTests(unittest.TestCase):
    def setUp(self):
        self.faces=np.array([[0,1,2],[3,4,5]])
        self.cross=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],
                             [0.,-.5,-1.],[0.,-.5,1.],[0.,.5,0.]])
    def test_cloth_crossing_and_transform(self):
        pairs,span=crossings(self.cross,self.faces)
        self.assertEqual(pairs.tolist(),[[0,1]])
        self.assertGreater(span[0],.1)
        moved=Rotation.from_euler("xyz",[31,12,-52],degrees=True).apply(self.cross)+[3.,-2.,1.]
        p,s=crossings(moved,self.faces)
        np.testing.assert_array_equal(p,pairs)
        np.testing.assert_allclose(s,span,atol=1e-10)
    def test_parallel_contact_is_not_crossing(self):
        points=np.r_[self.cross[:3],self.cross[:3]+[0.,0.,.0005]]
        self.assertEqual(len(crossings(points,self.faces)[0]),0)
    def test_shallow_crossing_separates_despite_large_normal_span(self):
        points = self.cross * .01
        points[3, 2] = -.0002
        pairs, spans = crossings(points, self.faces)
        self.assertGreater(spans[0], .001)
        shift = separating_translations(points, self.faces, pairs)[0]
        self.assertAlmostEqual(np.linalg.norm(shift), .0002, places=12)
        moved = points.copy()
        moved[3:] += shift * (1 + 1e-5)
        self.assertEqual(len(crossings(moved, self.faces)[0]), 0)
    def test_deep_crossing_still_exceeds_one_millimeter(self):
        points = self.cross * .004
        pairs, _ = crossings(points, self.faces)
        shift = separating_translations(points, self.faces, pairs)[0]
        self.assertGreater(np.linalg.norm(shift), .001)
        moved = points.copy()
        moved[3:] += shift * (1 + 1e-5)
        self.assertEqual(len(crossings(moved, self.faces)[0]), 0)
    def test_crossing_separation_is_rigid_transform_invariant(self):
        points = self.cross * .004
        pairs, _ = crossings(points, self.faces)
        shift = separating_translations(points, self.faces, pairs)
        rotation = Rotation.from_euler('xyz', [21, 37, -63], degrees=True)
        moved = rotation.apply(points) + [3., -1., 2.]
        transformed = separating_translations(moved, self.faces, pairs)
        np.testing.assert_allclose(np.linalg.norm(transformed, axis=1),
                                   np.linalg.norm(shift, axis=1), atol=1e-12)
        separated = moved.copy()
        separated[3:] += transformed[0] * (1 + 1e-5)
        self.assertEqual(len(crossings(separated, self.faces)[0]), 0)
    def test_plane_crossing_outside_triangle_is_not_collision(self):
        points=self.cross.copy()
        points[3:,0]+=4.
        self.assertEqual(len(crossings(points,self.faces)[0]),0)
    def test_sewn_neighbors_excluded(self):
        points=np.r_[self.cross[:3],[[0.,0.,1.]]]
        faces=np.array([[0,1,2],[1,0,3]])
        self.assertEqual(len(crossings(points,faces)[0]),0)
    def test_sewn_bend_pairs_are_only_seam_edges(self):
        from mesh import pocket_mesh,sewn_bend_pairs
        import json
        from pathlib import Path
        material=json.loads(Path(__file__).with_name("parameters.json").read_text())["estimated_probe"]
        points,faces,markers=pocket_mesh(material)
        pairs=sewn_bend_pairs(points,faces)
        self.assertEqual(len(pairs),58)
        for a,b in pairs:
            edge=set(faces[a])&set(faces[b])
            self.assertEqual(len(edge),2)
            self.assertTrue(edge.issubset(set(markers["upper"])&set(markers["base"])))
        moved=Rotation.from_euler("xyz",[25,60,-32],degrees=True).apply(points)+[1,2,3]
        np.testing.assert_array_equal(sewn_bend_pairs(moved,faces),pairs)

    def test_capsule_proxy_contains_true_capsule(self):
        from audit_mechanism import capsule_proxy
        from scipy.spatial import ConvexHull
        cap=dict(radius=.011,height=.478,center=[0.,0.,.25])
        vertices,error=capsule_proxy(cap)
        self.assertLess(error,.001)
        directions=np.random.default_rng(1).normal(size=(1000,3))
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        sphere=directions*cap["radius"]
        samples=np.r_[sphere+[0,0,.25-.239],sphere+[0,0,.25+.239]]
        eq=ConvexHull(vertices).equations
        self.assertLessEqual((samples@eq[:,:3].T+eq[:,3]).max(),1e-10)

    def test_cloth_crosses_gripper_even_with_vertices_outside(self):
        from robot_contacts import triangles_in_convex
        from scipy.spatial import ConvexHull
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        eq=ConvexHull(vertices).equations
        points=np.array([[-2.,-2.,0.],[2.,-2.,0.],[0.,2.,0.]])
        faces=np.array([[0,1,2]])
        self.assertEqual(triangles_in_convex(points,faces,eq),1)
        self.assertEqual(triangles_in_convex(points+[0,0,1.],faces,eq),0)

    def test_world_box_culling_preserves_triangle_interior_contacts(self):
        from robot_contacts import penetrating_gripper_shapes
        from scipy.spatial import ConvexHull
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        geometry=dict(body_names=['box'],colliders=[dict(body='box',name='test',
            points=vertices,planes=ConvexHull(vertices).equations)])
        points=np.array([[-2.,-2.,0.],[2.,-2.,0.],[0.,2.,0.]])
        faces=np.array([[0,1,2]])
        rng=np.random.default_rng(414)
        for _ in range(10):
            rotation=Rotation.random(random_state=rng)
            origin=rng.normal(size=3)
            pose=np.r_[origin,rotation.as_quat()][None]
            touching=rotation.apply(points)+origin
            clear=rotation.apply(points+[0,0,2.])+origin
            self.assertEqual(penetrating_gripper_shapes(touching,faces,pose,geometry),['test'])
            self.assertEqual(penetrating_gripper_shapes(clear,faces,pose,geometry),[])

    def test_station_translation_preserves_calibration(self):
        from station_coordinates import translated_rig,TRANSLATION_S_W
        from yam_arena import station
        stock=station.rig();shifted=translated_rig(stock,TRANSLATION_S_W)
        for side in ('left','right'):
            np.testing.assert_allclose(np.array(shifted.bases[side])-shifted.top_cam.pos,
                np.array(stock.bases[side])-stock.top_cam.pos,atol=1e-12)
            self.assertEqual(shifted.wrist_cams[side],stock.wrist_cams[side])
        self.assertEqual(shifted.top_cam.lens,stock.top_cam.lens)
        self.assertEqual(shifted.top_cam.rot_xyzw,stock.top_cam.rot_xyzw)
        self.assertEqual(shifted.dream_look,stock.dream_look)

    def test_convex_overlap_and_gap(self):
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        n,e=features(vertices)
        a=(np.zeros((1,3)),np.eye(3)[None],vertices,n,e)
        for offset,expected in [(1.2,0.),(.9,.1)]:
            b=(np.array([[offset,0.,0.]]),np.eye(3)[None],vertices,n,e)
            self.assertAlmostEqual(penetration(a,b)[0],expected,places=9)
    def test_convex_global_rotation_invariant(self):
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        n,e=features(vertices)
        rotation=Rotation.from_euler("xyz",[25,60,-32],degrees=True).as_matrix()
        offset=np.array([.9,0.,0.])
        origin=np.array([3.,-1.,2.])
        a=(origin[None],rotation[None],vertices,n,e)
        b=((origin+rotation@offset)[None],rotation[None],vertices,n,e)
        self.assertAlmostEqual(penetration(a,b)[0],.1,places=9)

    def test_handle_path_clearance_bound_and_planning_margin(self):
        from robot_clearance import prepare_convexes, capsule_contacts
        from scipy.spatial import ConvexHull
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        geometry=dict(colliders=[dict(name='box',body='body',points=vertices.tolist(),
                                     planes=ConvexHull(vertices).equations.tolist())])
        shapes=prepare_convexes(geometry)
        poses=dict(body=np.eye(4))
        hit=capsule_contacts(poses,shapes,np.array([.5002,0,0]),np.array([0,0,1]),.001,.2)
        self.assertAlmostEqual(hit[0]['depth_upper_bound_m'],.0008,places=12)
        self.assertFalse(capsule_contacts(poses,shapes,np.array([.506,0,0]),np.array([0,0,1]),.001,.2))
        self.assertTrue(capsule_contacts(poses,shapes,np.array([.506,0,0]),np.array([0,0,1]),.001,.2,margin=.01))

    def test_handle_path_clearance_is_frame_invariant(self):
        from robot_clearance import prepare_convexes, capsule_contacts
        from scipy.spatial import ConvexHull
        vertices=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        shapes=prepare_convexes(dict(colliders=[dict(name='box',body='body',points=vertices.tolist(),
                                                   planes=ConvexHull(vertices).equations.tolist())]))
        rotation=Rotation.from_euler('xyz',[23,49,-71],degrees=True).as_matrix()
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=[2,-3,1]
        hit=capsule_contacts(dict(body=pose),shapes,pose[:3,3]+rotation@np.array([.5002,0,0]),
                             rotation@np.array([0,0,1]),.001,.2)
        self.assertAlmostEqual(hit[0]['depth_upper_bound_m'],.0008,places=12)


if __name__=="__main__":
    unittest.main()
