"""CPU tests of sewn topology and the independent diagnostic's failure guards."""

import json
from pathlib import Path
import unittest

import numpy as np

from measurements import closed_pocket_faces, inside, pocket_seating, triangles_in_box
from mesh import audit, extrude_surface, pocket_mesh, split_lip, tet_boundary


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name("parameters.json").read_text())
        self.p = self.config["estimated_probe"]
        self.rest, self.faces, self.markers = pocket_mesh(self.p)
        self.closed = closed_pocket_faces(self.rest, self.faces, self.markers)

    def test_seams_are_manifold_and_connected(self):
        for spacing in (0.005, 0.0025):
            rest, faces, markers = pocket_mesh(self.p, spacing=spacing)
            result = audit(rest, faces)
            self.assertEqual(result["max_edge_multiplicity"], 2)
            self.assertTrue(result["connected"])
            self.assertEqual(len(markers["mouth_upper"]), len(markers["mouth_lower"]))

    def test_finite_seam_has_separate_flat_panels_and_closed_volume(self):
        from mesh import material_rest, edge_strain
        from robot_gate import fabric
        from scipy.spatial import cKDTree
        p = fabric(self.config, seam_height=.002)
        initial, faces, markers = pocket_mesh(p)
        rest = material_rest(initial, True, .002)
        self.assertTrue(audit(rest, faces)["connected"])
        self.assertGreaterEqual(cKDTree(rest).query(rest, k=2)[0][:, 1].min(), .002 - 1e-10)
        self.assertLess(abs(edge_strain(rest, initial, faces)).max(), .2)
        closed = closed_pocket_faces(rest, faces, markers)
        self.assertEqual(inside([[.04, 0, .003], [.04, 0, .005], [.10, 0, .003]], rest, closed).tolist(), [True, False, False])
        volume = abs(np.einsum("ij,ij->i", rest[closed[:, 0]], np.cross(rest[closed[:, 1]], rest[closed[:, 2]])).sum() / 6)
        self.assertAlmostEqual(volume, .075 * .14 * .002)

    def test_closed_volume_distinguishes_on_top_from_inside(self):
        found = inside([[0.05, 0, 0.008], [0.05, 0, 0.03], [0.1, 0, 0.008], [0.05, 0.1, 0.008]], self.rest, self.closed)
        self.assertEqual(found.tolist(), [True, False, False, False])
        on_top = pocket_seating(self.rest, self.closed, np.array([0.06, 0, 0.025]), [0.1, 0.1, 0.006])
        self.assertFalse(on_top["seated_samples"])

    def test_rotated_frame_end_uses_measured_orientation(self):
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler('y',10,degrees=True).as_matrix()
        moved=self.rest@rotation.T+[.2,-.1,.3]
        centre=np.array([.06,0.,.006])
        baseline=pocket_seating(self.rest,self.closed,centre,[.1,.1,.006],profile='beveled')
        rotated=pocket_seating(moved,self.closed,rotation@centre+[.2,-.1,.3],[.1,.1,.006],profile='beveled',rotation=rotation)
        self.assertEqual(rotated['inside_fraction'],baseline['inside_fraction'])
        self.assertTrue(baseline['seated_samples'])

    def test_triangle_passage_caught_when_vertices_are_outside(self):
        points = np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0, 0.02, 0]])
        size = [0.01, 0.01, 0.006]
        self.assertTrue(np.all(np.any(np.abs(points) > np.array(size) / 2, axis=1)))
        self.assertEqual(triangles_in_box(points, [[0, 1, 2]], np.zeros(3), size), 1)
        self.assertEqual(triangles_in_box(points + [0, 0, 0.01], [[0, 1, 2]], np.zeros(3), size), 0)

    def test_wrong_cover_orientation_is_rejected(self):
        flipped = self.rest.copy()
        flipped[:, 1:] *= -1
        result = pocket_seating(flipped, self.closed, np.array([0.06, 0, -0.008]), [0.1, 0.1, 0.006], markers=self.markers, faces=self.faces)
        self.assertFalse(result["orientation_correct"])
        self.assertFalse(result["seated_samples"])

    def test_volume_lip_has_positive_tets_and_closed_boundary(self):
        parts, _ = split_lip(self.rest, self.faces, self.markers)
        _, surface, faces = parts[1]
        nodes, tets, middle = extrude_surface(surface, faces, 0.0015)
        np.testing.assert_array_equal(nodes[middle], surface)
        corners = nodes[tets]
        det = np.linalg.det(corners[:, 1:] - corners[:, :1])
        self.assertTrue(np.all(det > 0))
        boundary = tet_boundary(tets)
        used, remap = np.unique(boundary, return_inverse=True)
        self.assertTrue(audit(nodes[used], remap.reshape(-1, 3))["connected"])
        # A thin extrusion must preserve the source surface area times thickness.
        area = np.linalg.norm(np.cross(surface[faces[:, 1]] - surface[faces[:, 0]],
                                       surface[faces[:, 2]] - surface[faces[:, 0]]), axis=1).sum() / 2
        self.assertAlmostEqual(float(det.sum() / 6 / (area * 0.0015)), 1.0, delta=0.01)

    def test_lip_partition_preserves_all_faces_and_only_sews_shared_nodes(self):
        parts, shared = split_lip(self.rest, self.faces, self.markers)
        reconstructed = []
        for ids, points, faces in parts:
            audit(points, faces)
            np.testing.assert_allclose(points, self.rest[ids])
            reconstructed.extend(ids[faces].tolist())
        self.assertEqual(sorted(map(tuple, reconstructed)), sorted(map(tuple, self.faces.tolist())))
        self.assertGreater(len(shared), 20)



class BeveledToolTests(unittest.TestCase):
    def test_cutaway_is_air_but_crossing_the_plate_is_penetration(self):
        from blade_geometry import node_penetration, triangle_penetration
        size=np.array([.1,.1,.006])
        self.assertEqual(node_penetration(np.array([[-.049,0.,0.]]),np.zeros(3),size,"beveled"),0.)
        crossing=np.array([[-.08,-.08,0.],[.08,-.08,0.],[0.,.08,0.]])
        self.assertEqual(triangle_penetration(crossing,np.array([[0,1,2]]),np.zeros(3),size,"beveled"),1)
        cutaway=np.array([[-.049,-.01,.001],[-.049,.01,.001],[-.047,0.,.001]])
        self.assertEqual(triangle_penetration(cutaway,np.array([[0,1,2]]),np.zeros(3),size,"beveled"),0)

    def test_authored_hull_and_scoring_halfspaces_agree(self):
        from blade_geometry import shape,planes
        from scipy.spatial import ConvexHull
        size=np.array([.1,.1,.006])
        vertices,_=shape(size)
        normals,offsets=planes(size,"beveled")
        hull=ConvexHull(vertices)
        points=np.random.default_rng(18).uniform(-size,size,(10000,3))
        authored=np.all(points@hull.equations[:,:3].T+hull.equations[:,3]<=1e-10,axis=1)
        assessed=np.all(points@normals.T<=offsets+1e-10,axis=1)
        self.assertTrue(np.array_equal(authored,assessed))

if __name__ == "__main__":
    unittest.main()
