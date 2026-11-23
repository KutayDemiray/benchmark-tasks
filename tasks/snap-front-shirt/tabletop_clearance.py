"""Conservative geometric bounds for complete robot withdrawal from the shirt."""

import numpy as np

from robot_geometry import rotation


def robot_clearance(row,nodes,meta,bounds):
    """Bound full triangles and hardware away from enclosing robot link boxes.

    Every box encloses all stock collision hull vertices of its link. Signed
    distance to that box is a conservative lower bound on separation from the
    real link. Subtract the longest cloth edge to cover triangle interiors,
    and a 9 mm enclosing radius for every complete snap half.
    """
    nodes=np.asarray(nodes,dtype=float)
    edges=np.asarray(meta['cloth']['edges'])
    cover=float(np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1).max())
    centers=np.array([p[:3] for name,p in row['physics']['hardware'].items() if '_ball' not in name])
    cloth_bound=hardware_bound=float('inf')
    for side in ('left','right'):
        for name,pose in zip(row[side+'_link_names'],row[side+'_link_poses_xyzw'],strict=True):
            R=rotation(pose[3:]);t=np.asarray(pose[:3])
            if name not in bounds:continue
            local=(nodes-t)@R;hardware=(centers-t)@R
            low,high=np.asarray(bounds[name]);center=(low+high)/2;half=(high-low)/2
            def signed(points):
                q=np.abs(points-center)-half
                return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
            cloth_bound=min(cloth_bound,float(signed(local).min())-cover)
            hardware_bound=min(hardware_bound,float(signed(hardware).min())-.009)
    return dict(fabric_m=cloth_bound,hardware_m=hardware_bound)


def stock_collision_bounds():
    """Read the installed stock collision geometry without changing it."""
    from yam_arena.analysis import link_hulls
    bounds={}
    for name,hulls in link_hulls().items():
        vertices=np.concatenate([h.vertices for h in hulls])
        bounds[name]=np.array([vertices.min(axis=0),vertices.max(axis=0)])
    return bounds
