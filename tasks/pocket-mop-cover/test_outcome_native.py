"""Counterexamples for conservative native support and clearance assessment."""

import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from outcome_native import NativeOutcome,cloth_clearance,gap,shape_clearance,swept_shapes


class NativeClearanceTests(unittest.TestCase):
    def test_parallel_slanted_shapes_do_not_gain_false_support_from_their_boxes(self):
        vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))*[.1,.005,.005]
        rotation=Rotation.from_euler('z',45.,degrees=True)
        offset=rotation.apply([0.,.04,0.])
        first=np.array([[[0.,0.,0.,*rotation.as_quat()]]])
        second=np.array([[[*offset,*rotation.as_quat()]]])
        geometry=[dict(body='part',points=vertices)]
        a=swept_shapes(geometry,first,['part'])
        b=swept_shapes(geometry,second,['part'])
        self.assertEqual(gap(a[0]['low'],a[0]['high'],b[0]['low'],b[0]['high']),0.)
        self.assertAlmostEqual(shape_clearance(a,b),.030,places=8)

    def test_midinterval_rotation_contact_cannot_be_certified_clear(self):
        corners=np.array(list(itertools.product((-1.,1.),repeat=3)))
        geometry=[dict(body='part',points=corners*[.1,.005,.005])]
        rotations=Rotation.from_euler('z',[[0.],[90.],[0.]],degrees=True).as_quat()
        poses=np.array([[[0.,.04,0.,*q]] for q in rotations])
        obstacle=swept_shapes([dict(body='part',points=corners*.005)],
                             np.array([[[0.,0.,0.,0.,0.,0.,1.]]]),['part'])
        self.assertGreater(shape_clearance(swept_shapes(geometry,poses[[0,-1]],['part']),obstacle),.02)
        self.assertEqual(shape_clearance(swept_shapes(geometry,poses,['part']),obstacle),0.)

    def test_arbitrary_overlapping_convex_shapes_never_get_positive_separation(self):
        rng=np.random.default_rng(31)
        vertices=np.array(list(itertools.product((-.01,.01),repeat=3)))
        geometry=[dict(body='part',points=vertices)]
        for _ in range(40):
            shapes=[]
            for _ in range(2):
                rotation=Rotation.random(random_state=rng)
                # Both cubes contain the world origin, regardless of rotation.
                center=rotation.apply(rng.uniform(-.009,.009,3))
                shapes.append(swept_shapes(geometry,np.array([[[*center,*rotation.as_quat()]]]),['part']))
            self.assertEqual(shape_clearance(*shapes),0.)

    def test_triangle_interior_cannot_be_certified_free_from_vertex_distance(self):
        points=np.array([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.]])
        shapes=[dict(low=np.array([-.01,-.01,-.01]),high=np.array([.01,.01,.01]))]
        self.assertGreater(np.linalg.norm(points,axis=1).min(),.1)
        self.assertEqual(cloth_clearance(points[None],np.array([[0,1,2]]),shapes),0.)

    def test_midinterval_contact_prevents_a_false_release(self):
        vertices=np.array(list(itertools.product((-.005,.005),repeat=3)))
        geometry=[dict(body='hand',points=vertices)]
        poses=np.array([[[0.,0.,.1,0.,0.,0.,1.]],[[0.,0.,0.,0.,0.,0.,1.]],[[0.,0.,.1,0.,0.,0.,1.]]])
        object_bounds=[dict(low=np.array([-.01,-.01,-.01]),high=np.array([.01,.01,.01]))]
        endpoints=swept_shapes(geometry,poses[[0,-1]],['hand'])
        self.assertGreater(shape_clearance(endpoints,object_bounds),.02)
        self.assertEqual(shape_clearance(swept_shapes(geometry,poses,['hand']),object_bounds),0.)

    def test_box_gap_does_not_exaggerate_diagonal_clearance(self):
        self.assertAlmostEqual(float(gap(np.array([0.,0.,0.]),np.array([1.,1.,1.]),
            np.array([1.03,1.04,0.]),np.array([2.,2.,1.]))),.05)


class NativePocketScoringTests(unittest.TestCase):
    """Geometry fixtures exercise observation wiring, not simulated feasibility."""

    def setUp(self):
        from cover_mesh import cover_mesh
        from robot_gate import fabric
        config=json.loads(Path(__file__).with_name('parameters.json').read_text())
        self.points,faces,cover=cover_mesh(fabric(config))
        corners=np.array(list(itertools.product((-.5,.5),repeat=3)))
        frame={'colliders':[
            dict(body='Left',points=corners*[.1,.1,.006]+[-.142,0,0]),
            dict(body='Right',points=corners*[.1,.1,.006]+[.142,0,0]),
            dict(body='Handle',points=corners*[.02,.02,.10])]}
        names=dict(body_names=['Left','Right','Bridge','Handle'],joint_names=['HingeLeft','HingeRight'])
        robots={side:dict(body_names=['gripper'],colliders=[dict(body='gripper',points=corners*.02)])
                for side in ('left','right')}
        self.observer=NativeOutcome(self.points,faces,cover,frame,names,robots,0.,480,{},([-1.,-1.],[1.,1.]))
        poses=np.array([[.212,0,.006,0,0,0,1],[.228,0,.006,0,0,0,1],
                        [.22,0,.006,0,0,0,1],[.25,0,.1,0,0,0,1]])
        self.state=dict(positions=np.repeat(self.points[None],24,0),velocities=np.zeros((24,len(self.points),3)),
            frame_body_poses=np.repeat(poses[None],24,0),frame_joint_positions=np.zeros((24,2)),
            left_body_poses=np.repeat(np.array([[[1.,1.,1.,0.,0.,0.,1.]]]),24,0),
            right_body_poses=np.repeat(np.array([[[-1.,-1.,1.,0.,0.,0.,1.]]]),24,0))

    def test_actual_one_pocket_geometry_gets_only_quarter_credit(self):
        self.state['frame_body_poses'][:,1,2]+=.04
        for step in range(40):observation,result=self.observer.update(step,self.state)
        self.assertEqual(observation.pocket_depth_m,(.04,0.))
        self.assertEqual(result['score'],.25)
        self.assertFalse(result['retention_verified'])

    def test_frame_resting_on_top_has_no_enclosure_credit(self):
        self.state['frame_body_poses'][:,:2,2]+=.04
        observation,result=self.observer.update(0,self.state)
        self.assertEqual(observation.pocket_depth_m,(0.,0.))
        self.assertEqual(result['score'],0.)

    def test_middle_sheet_wrinkles_preserve_pocket_credit(self):
        # The loose middle sheet can wrinkle without changing either pocket.
        middle=(self.points[:,0]>.1)&(self.points[:,0]<.34)
        phase=(self.points[middle,0]-.1)/.24
        self.state['positions'][:,middle,2]+=.008*np.sin(np.pi*phase)**2
        self.observer.previous=self.state['positions'][0].copy()
        for step in range(40):
            observation,result=self.observer.update(step,self.state)
        self.assertEqual(observation.pocket_depth_m,(.04,.04))
        self.assertEqual(result['score'],.5)
        self.assertFalse(result['retention_verified'])

    def test_common_translation_and_yaw_preserve_pocket_credit(self):
        # Changing the assembly's position in a view is irrelevant to seating.
        rotation=Rotation.from_euler('z',73.,degrees=True)
        translation=np.array([.1,-.2,0.])
        points=self.state['positions']
        points[:]=rotation.apply(points.reshape(-1,3)).reshape(points.shape)+translation
        for name in ('frame_body_poses','left_body_poses','right_body_poses'):
            poses=self.state[name]
            poses[:,:,:3]=rotation.apply(poses[:,:,:3].reshape(-1,3)).reshape(poses[:,:,:3].shape)+translation
            orientations=rotation*Rotation.from_quat(poses[:,:,3:].reshape(-1,4))
            poses[:,:,3:]=orientations.as_quat().reshape(poses[:,:,3:].shape)
        self.observer.previous=points[0].copy()
        for step in range(40):
            observation,result=self.observer.update(step,self.state)
        self.assertEqual(observation.pocket_depth_m,(.04,.04))
        self.assertEqual(result['score'],.5)
        self.assertFalse(result['retention_verified'])
