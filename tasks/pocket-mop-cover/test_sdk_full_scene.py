"""Check canonical state ordering and exact native convex-hull interpretation on CPU."""

import itertools
import struct
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from full_sdk import NativeView
from read_native_properties import decode_value
from sdk_cooked_geometry import hulls


class NativeStateTests(unittest.TestCase):
    def test_native_body_and_joint_permutations_are_independent(self):
        class Simulator:
            def bind(self,kind,path):
                return SimpleNamespace(kind=kind,body_names=['B','A'],dof_names=['z','x','y'])
            def read(self,binding):
                if 'DOF' in binding.kind:return np.array([[30.,10.,20.]])
                return np.array([[200.,100.]])
        view=NativeView(Simulator(),'/Frame',['A','B'],['x','y','z'])
        np.testing.assert_array_equal(view.get_link_transforms(),[[100.,200.]])
        np.testing.assert_array_equal(view.get_masses(),[[100.,200.]])
        np.testing.assert_array_equal(view.get_dof_positions(),[[10.,20.,30.]])
        np.testing.assert_array_equal(view.get_dof_dampings(),[[10.,20.,30.]])
        with self.assertRaises(AttributeError):view.set_dof_positions
        with self.assertRaises(AssertionError):NativeView(Simulator(),'/Frame',['A'],['x','y','z'])

    def test_large_native_vertex_arrays_decode_only_when_explicitly_requested(self):
        values=list(range(90));raw=struct.pack('<90f',*values)
        self.assertIsInstance(decode_value(raw,{'type':8}),dict)
        self.assertEqual(decode_value(raw,{'type':8},unlimited_array=True),values)

    def test_native_mesh_scale_and_shape_pose_are_applied_in_link_frame(self):
        points=np.array(list(itertools.product((-.01,.01),repeat=3)))
        rotation=Rotation.from_euler('z',90,degrees=True)
        scaled=rotation.apply(points*[2.,3.,4.])+[.1,.2,.3]
        authored={};reference={};objects=[]
        def obj(handle,kind,props,events=()):
            return dict(handle=handle,context=1,class_name=kind,
                properties={key:dict(first=value,last=value,updates=1) for key,value in props.items()},list_events=list(events))
        for offset,side,root in ((0,'left','LeftArm'),(10,'right','RightArm')):
            item=dict(body='gripper',name='/nested/gripper/Shape',points=scaled.tolist())
            authored[side]=reference[side]=dict(body_names=['gripper'],indices=[0],colliders=[item],quaternion_order='xyzw')
            objects.extend([
                obj(offset+1,'PxArticulationLink',{'name':'/World/envs/env_0/'+root+'/base/link5/gripper'},
                    [dict(command=7,property='shapes',value=offset+2)]),
                obj(offset+2,'PxShape',{'shapeFlags':{'value':1},'geom':offset+3,
                    'localPose':[*rotation.as_quat(),.1,.2,.3],'contactOffset':.0005,'restOffset':0.}),
                obj(offset+3,'PxConvexMeshGeometry',{'scale':[2.,3.,4.],'convexMesh':offset+4}),
                obj(offset+4,'PxConvexMesh',{'verts':points.ravel().tolist()})])
        result,comparison=hulls({'objects':objects},authored,reference)
        self.assertLess(max(item['hausdorff_distance_m'] for item in comparison),1e-12)
        for side in ('left','right'):
            actual=np.asarray(result[side]['colliders'][0]['points'])
            self.assertEqual(len(actual),8)
            np.testing.assert_allclose(actual.min(0),scaled.min(0))
            np.testing.assert_allclose(actual.max(0),scaled.max(0))
            self.assertEqual(result[side]['geometry_source'],'native_sdk_omnipvd')
        wrong={'objects':objects+[objects[0]]}
        with self.assertRaisesRegex(AssertionError,'reused'):
            hulls(wrong,authored,reference)


if __name__=='__main__':
    unittest.main()
