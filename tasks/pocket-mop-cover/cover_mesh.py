"""Connected two-pocket cover geometry; no simulation attachments or guides.

This is geometry preparation for the next isolated integration test. It does not
qualify the material or frame and is not a full benchmark environment.
"""
import numpy as np
from mesh import pocket_mesh,audit
from measurements import closed_pocket_faces


def cover_mesh(material,spacing=.005,mouth_height=.016,length=.44):
    """Join two sewn coupon halves along their common middle sheet boundary."""
    half=dict(material,coupon_length_m=length/2)
    local,local_faces,local_markers=pocket_mesh(half,spacing=spacing,mouth_height=mouth_height)
    local_closed=closed_pocket_faces(local,local_faces,local_markers)
    points=local.tolist()
    left_map=np.arange(len(local),dtype=np.int32)
    middle={round(float(local[i,1]),10):i for i in local_markers['base_grid'][-1]}
    right_map=[]
    for point in local:
        mirrored=np.array([length-point[0],-point[1],point[2]])
        if abs(point[0]-length/2)<1e-10:
            right_map.append(middle[round(float(mirrored[1]),10)])
        else:
            right_map.append(len(points));points.append(mirrored.tolist())
    right_map=np.asarray(right_map,dtype=np.int32)
    faces=np.r_[local_faces,right_map[local_faces]]
    pockets=[]
    for mapping in (left_map,right_map):
        markers={key:mapping[local_markers[key]].tolist() for key in
            ('mouth_upper','mouth_lower','upper','base','upper_grid','base_grid')}
        markers.update(dimensions_m=local_markers['dimensions_m'],seam_height_m=local_markers['seam_height_m'])
        pockets.append(dict(markers=markers,faces=mapping[local_faces].tolist(),closed_faces=mapping[local_closed].tolist()))
    points=np.asarray(points)
    audit(points,faces)
    return points,faces,dict(length_m=length,width_m=material['coupon_width_m'],pockets=pockets,
        middle_join_node_ids=sorted(middle.values()),construction='One continuous base and two three-sided sewn pocket panels')
