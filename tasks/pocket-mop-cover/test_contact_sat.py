"""Check optimized independent contact audit against exhaustive SAT."""
import itertools
import unittest
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from audit_mechanism import features,penetration,capsule_proxy


def exhaustive(points):
    hull=ConvexHull(points)
    edges=set()
    for face in hull.simplices:
        edges.update(tuple(sorted(pair)) for pair in itertools.combinations(face,2))
    directions=np.array([points[b]-points[a] for a,b in edges])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    return hull.equations[:,:3],directions


def reference(pa,ra,va,pb,rb,vb):
    na,ea=exhaustive(va);nb,eb=exhaustive(vb)
    axes=np.r_[na@ra.T,nb@rb.T,np.cross((ea@ra.T)[:,None],(eb@rb.T)[None]).reshape(-1,3)]
    axes=axes[np.linalg.norm(axes,axis=1)>1e-8]
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    sa=(va@ra.T+pa)@axes.T;sb=(vb@rb.T+pb)@axes.T
    return max(0.,float(np.minimum(sa.max(0)-sb.min(0),sb.max(0)-sa.min(0)).min()))


class ContactSatTests(unittest.TestCase):
    def test_capsule_projection_is_invariant_under_common_rigid_motion(self):
        from audit_frame_robot import prepare,capsule_projection_bound
        cube=np.array(list(itertools.product((-1.,1.),repeat=3)))*[.01,.02,.03]
        centers=np.array([[.0202,0.,.25],[.019,0.,.25],[.022,0.,.25]])
        for rotation,translation in [(np.eye(3),np.zeros(3)),
                (Rotation.from_euler('xyz',[34,65,27],degrees=True).as_matrix(),np.array([.3,-.2,.6]))]:
            quaternion=Rotation.from_matrix(rotation).as_quat()
            cap_poses=np.tile(np.r_[translation,quaternion],(3,1))
            box_poses=np.c_[centers@rotation.T+translation,np.tile(quaternion,(3,1))]
            capsule=prepare(dict(body='Handle',capsule=dict(radius=.011,height=.478,center=[0.,0.,.25])),cap_poses)
            box=prepare(dict(body='Finger',points=cube),box_poses)
            bound=capsule_projection_bound(capsule,box,np.arange(3))
            np.testing.assert_allclose(bound,[.0008,.002,0.],atol=1e-12)

    def test_capsule_refinement_preserves_real_penetration_failure(self):
        from audit_frame_robot import prepare,compare
        cube=np.array(list(itertools.product((-1.,1.),repeat=3)))*[.01,.02,.03]
        poses=np.tile([0.,0.,0.,0.,0.,0.,1.],(2,1))
        capsule=prepare(dict(body='Handle',capsule=dict(radius=.011,height=.478,center=[0.,0.,.25])),poses)
        poses=poses.copy();poses[:,:3]=[[.0202,0.,.25],[.019,0.,.25]]
        box=prepare(dict(body='Finger',points=cube),poses)
        depth,violations,refinement=compare(capsule,box)
        self.assertAlmostEqual(depth,.002)
        self.assertEqual(violations,[1])
        self.assertIsNotNone(refinement)

    def test_rotated_intersections_match_exhaustive_axes(self):
        rng=np.random.default_rng(71)
        cube=np.array(list(itertools.product((-.02,.02),repeat=3)))
        poly=rng.normal(size=(15,3))*.025
        pa=rng.uniform(-.06,.06,(30,3));pb=np.zeros((30,3))
        ra=Rotation.random(30,random_state=rng).as_matrix()
        rb=Rotation.random(30,random_state=rng).as_matrix()
        measured=penetration((pa,ra,cube,*features(cube)),(pb,rb,poly,*features(poly)))
        expected=[reference(pa[i],ra[i],cube,pb[i],rb[i],poly) for i in range(30)]
        np.testing.assert_allclose(measured,expected,atol=1e-10)
        self.assertGreater(np.count_nonzero(measured),0)
        self.assertGreater(np.count_nonzero(measured==0),0)

    def test_capsule_bound_matches_exhaustive_axes(self):
        capsule=capsule_proxy(dict(radius=.0125,height=.475,center=[0.,0.,.25]))[0]
        box=np.array(list(itertools.product((-.01,.01),repeat=3)))
        pa=np.zeros((3,3));pb=np.array([[.02,0.,.25],[.024,0.,.25],[0.,0.,.51]])
        rotations=np.tile(np.eye(3),(3,1,1))
        measured=penetration((pa,rotations,capsule,*features(capsule)),(pb,rotations,box,*features(box)))
        expected=[reference(pa[i],rotations[i],capsule,pb[i],rotations[i],box) for i in range(3)]
        np.testing.assert_allclose(measured,expected,atol=1e-10)
        self.assertLess(measured[1],.001)

    def test_threshold_is_a_conservative_bound(self):
        rng=np.random.default_rng(72)
        a=rng.normal(size=(18,3))*.02;b=rng.normal(size=(16,3))*.02
        pa=rng.uniform(-.05,.05,(80,3));pb=np.zeros_like(pa)
        ra=Rotation.random(80,random_state=rng).as_matrix();rb=Rotation.random(80,random_state=rng).as_matrix()
        aa=(pa,ra,a,*features(a));bb=(pb,rb,b,*features(b))
        exact=penetration(aa,bb);bound=penetration(aa,bb,.001)
        self.assertTrue(np.all(bound>=exact-1e-12))
        np.testing.assert_array_equal(bound>.001,exact>.001)
