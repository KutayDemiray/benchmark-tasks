"""Compare refitted broad-phase candidates against exhaustive geometry."""

import itertools
import unittest

import numpy as np

from self_contact import crossings
from triangle_broadphase import TriangleBroadphase


def pair_set(pairs):
    return {tuple(pair) for pair in pairs}


class TriangleBroadphaseTests(unittest.TestCase):
    def test_arbitrary_deformation_preserves_every_box_pair(self):
        rng = np.random.default_rng(8917)
        initial = rng.normal(size=(90, 3))
        faces = np.array([rng.choice(90, 3, replace=False) for _ in range(60)])
        tree = TriangleBroadphase(initial, faces, leaf_size=3)
        for _ in range(5):
            # A fixed hierarchy must remain valid even after unrelated motion.
            triangles = rng.normal(size=initial.shape)[faces]
            low, high = triangles.min(1), triangles.max(1)
            expected = set()
            for i, j in itertools.combinations(range(len(faces)), 2):
                if set(faces[i]) & set(faces[j]):
                    continue
                if np.all(high[i] >= low[j]) and np.all(high[j] >= low[i]):
                    expected.add((i, j))
            actual = tree.candidates(triangles)
            self.assertEqual(pair_set(actual), expected)
            self.assertEqual(len(actual), len(expected))

    def test_reference_crossings_and_depths_match(self):
        rng = np.random.default_rng(8021)
        points = rng.normal(size=(120, 3))
        faces = np.arange(120).reshape(-1, 3)
        tree = TriangleBroadphase(points, faces)
        for _ in range(3):
            points = rng.normal(size=points.shape)
            reference, reference_depth = crossings(points, faces)
            accelerated, accelerated_depth = crossings(points, faces, tree)
            self.assertEqual(pair_set(reference), pair_set(accelerated))
            expected = dict(zip(map(tuple, reference), reference_depth))
            for pair, depth in zip(accelerated, accelerated_depth):
                self.assertAlmostEqual(depth, expected[tuple(pair)], places=14)

    def test_disjoint_and_touching_bounds(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 0., 0.], [2., 0., 0.], [1., 1., 0.],
                           [3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])
        faces = np.arange(9).reshape(-1, 3)
        tree = TriangleBroadphase(points, faces, leaf_size=1)
        self.assertEqual(pair_set(tree.candidates(points[faces])), {(0, 1)})
        self.assertEqual(len(crossings(points, faces, tree)[0]), 0)


if __name__ == '__main__':
    unittest.main()
