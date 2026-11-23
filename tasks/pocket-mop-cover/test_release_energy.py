"""Check the mechanical accounting independently of any favorable task outcome."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_release_energy import SurfaceEnergy, rigid_energy


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        self.material=dict(density_kg_m3=1000.,shell_thickness_m=.001,
                           young_modulus_pa=2e6,poisson_ratio=.3,bend_modulus_pa=2e4)
        self.energy=SurfaceEnergy(self.points,np.array([[0,1,2],[0,2,3]]),self.material)

    def test_rest_elastic_energy_is_zero_under_rigid_motion(self):
        moved=Rotation.from_euler('xyz',[34.,-12.,81.],degrees=True).apply(self.points)+[3.,-2.,1.]
        values=self.energy.measure(moved,np.zeros_like(moved))
        self.assertLess(values['membrane_proxy_j'],1e-20)
        self.assertLess(values['bending_proxy_j'],1e-20)
        self.assertAlmostEqual(self.energy.node_mass.sum(),1.)

    def test_cloth_translation_work_and_kinetic_energy(self):
        before=self.energy.measure(self.points,np.zeros_like(self.points))
        after=self.energy.measure(self.points+[0.,0.,.1],np.tile([2.,0.,0.],(4,1)))
        self.assertAlmostEqual(after['cloth_gravity_j']-before['cloth_gravity_j'],.981)
        self.assertAlmostEqual(after['cloth_kinetic_j'],2.)

    def test_bending_does_not_change_total_material_mass(self):
        folded=self.points.copy();folded[3]=[0.,0.,1.]
        values=self.energy.measure(folded,np.zeros_like(folded))
        self.assertGreater(values['bending_proxy_j'],0.)
        self.assertAlmostEqual(self.energy.node_mass.sum(),1.)

    def test_rigid_energy_uses_com_offset_inertia_and_clip_spring(self):
        dynamics=dict(masses_kg=[[2.]],local_com_poses_xyzw=[[[0.,0.,.5,0.,0.,0.,1.]]],
                      inertias_kg_m2=[[[1.,0.,0.,0.,2.,0.,0.,0.,3.]]],
                      joint_names=['ClipFlexure'],joint_stiffnesses=[[4.]])
        pose=np.array([[0.,0.,1.,0.,0.,0.,1.]])
        values=rigid_energy(pose,np.array([[2.,0.,0.,0.,0.,2.]]),np.array([.5]),dynamics)
        self.assertAlmostEqual(values['rigid_gravity_j'],2*9.81*1.5)
        self.assertAlmostEqual(values['rigid_kinetic_j'],10.)
        self.assertAlmostEqual(values['clip_spring_j'],.5)
