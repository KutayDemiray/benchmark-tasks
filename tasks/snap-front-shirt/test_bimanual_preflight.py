"""Check conservative separation bounds for simple enclosing boxes."""

import unittest
import numpy as np
from bimanual_preflight import separation


class SeparationChecks(unittest.TestCase):
    def test_separated_boxes_have_the_known_axis_gap(self):
        first=('a',np.zeros(3),np.eye(3),np.full(3,.5))
        second=('b',np.array([2.,0.,0.]),np.eye(3),np.full(3,.5))
        self.assertAlmostEqual(separation(first,second),1.)
        self.assertAlmostEqual(separation(second,first),1.)

    def test_overlapping_boxes_cannot_pass_the_endpoint_screen(self):
        angle=np.pi/4
        rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        first=('a',np.zeros(3),np.eye(3),np.full(3,.5))
        second=('b',np.array([.1,.1,0.]),rotation,np.full(3,.5))
        self.assertLess(separation(first,second),0.)
