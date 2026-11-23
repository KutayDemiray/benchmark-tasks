"""Geometric diagnostics for crossings between nonadjacent cloth triangles."""
import numpy as np
from scipy.spatial import cKDTree


def separating_translations(points, faces, pairs):
    """Return a separating translation of the second triangle in each pair.

    Each candidate moves one triangle entirely to one side of the other's
    plane. The shortest of these four translations is an upper bound on the
    minimum separating distance, even though edge axes are not searched.
    Input pairs must be actual crossings returned by ``crossings``. A tiny
    additional move in the returned direction makes separation strict.
    """
    if not len(pairs):
        return np.empty((0, 3))
    triangles = np.asarray(points, dtype=np.float64)[faces]
    a, b = triangles[pairs[:, 0]], triangles[pairs[:, 1]]
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    na /= np.linalg.norm(na, axis=1)[:, None]
    nb /= np.linalg.norm(nb, axis=1)[:, None]
    db = np.einsum("npc,nc->np", b - a[:, None, 0], na)
    da = np.einsum("npc,nc->np", a - b[:, None, 0], nb)
    shifts = np.stack((-db.min(1)[:, None] * na, -db.max(1)[:, None] * na,
                       da.min(1)[:, None] * nb, da.max(1)[:, None] * nb), axis=1)
    selected = np.linalg.norm(shifts, axis=2).argmin(1)
    return shifts[np.arange(len(pairs)), selected]


def crossings(points, faces, broadphase=None):
    """Return crossing face pairs and their conservative normal-span estimate.

    The span is a diagnostic severity estimate, not a closest-distance measure.
    Parallel contact and triangles sharing a material vertex are excluded.
    """
    triangles = np.asarray(points)[faces]
    centers = triangles.mean(1)
    radii = np.linalg.norm(triangles-centers[:,None],axis=2).max(1)
    pairs = (cKDTree(centers).query_pairs(2*radii.max(),output_type="ndarray")
             if broadphase is None else broadphase.candidates(triangles))
    if not len(pairs):
        return pairs, np.empty(0)
    i,j=pairs.T
    keep = np.linalg.norm(centers[i]-centers[j],axis=1) < radii[i]+radii[j]
    lower,upper=triangles.min(1),triangles.max(1)
    keep &= np.all(upper[i]>=lower[j],axis=1)&np.all(upper[j]>=lower[i],axis=1)
    keep &= ~np.any(faces[i,:,None] == faces[j,None,:],axis=(1,2))
    pairs=pairs[keep]
    i,j=pairs.T
    a,b=triangles[i],triangles[j]
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
    nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    la,lb=np.linalg.norm(na,axis=1),np.linalg.norm(nb,axis=1)
    assert np.all(la>1e-12) and np.all(lb>1e-12), "Degenerate material face"
    na/=la[:,None]
    nb/=lb[:,None]
    db=np.einsum("npc,nc->np",b-a[:,None,0],na)
    da=np.einsum("npc,nc->np",a-b[:,None,0],nb)
    keep=(db.min(1)<-1e-7)&(db.max(1)>1e-7)&(da.min(1)<-1e-7)&(da.max(1)>1e-7)
    a,b,da,db,pairs=a[keep],b[keep],da[keep],db[keep],pairs[keep]
    if not len(pairs):
        return pairs,np.empty(0)

    def edges_hit(triangle,other,distance):
        u=triangle[:,1]-triangle[:,0]
        v=triangle[:,2]-triangle[:,0]
        uu=np.einsum("nc,nc->n",u,u)
        uv=np.einsum("nc,nc->n",u,v)
        vv=np.einsum("nc,nc->n",v,v)
        denominator=uu*vv-uv*uv
        hit=np.zeros(len(triangle),bool)
        for k in range(3):
            j=(k+1)%3
            crosses=distance[:,k]*distance[:,j]<0
            difference=distance[:,k]-distance[:,j]
            alpha=distance[:,k]/np.where(abs(difference)>1e-12,difference,1.)
            point=other[:,k]+alpha[:,None]*(other[:,j]-other[:,k])-triangle[:,0]
            pu=np.einsum("nc,nc->n",point,u)
            pv=np.einsum("nc,nc->n",point,v)
            aa=(pu*vv-pv*uv)/denominator
            bb=(pv*uu-pu*uv)/denominator
            hit |= crosses&(aa>=-1e-6)&(bb>=-1e-6)&(aa+bb<=1+1e-6)
        return hit

    hit=edges_hit(a,b,db)|edges_hit(b,a,da)
    span=np.maximum(np.minimum(db.max(1),-db.min(1)),np.minimum(da.max(1),-da.min(1)))
    return pairs[hit],span[hit]
