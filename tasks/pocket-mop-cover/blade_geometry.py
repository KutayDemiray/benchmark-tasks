"""Estimated bevel of the visible plastic frame end, not a cloth guide."""

import numpy as np


def planes(size, profile="square"):
    half = np.asarray(size) / 2
    normals = [*np.eye(3), *(-np.eye(3))]
    offsets = [*half, *half]
    if profile == "beveled":
        # 12 mm long top bevel, 1 mm nose, 6 mm main plate. Dimensions are
        # explicit estimates from the manufacturer's low-profile frame images.
        slope = (size[2] - .001) / .012
        normal = np.array([-slope, 0., 1.])
        offset = -half[2] + .001 + slope*half[0]
        scale = np.linalg.norm(normal)
        normals.append(normal/scale)
        offsets.append(offset/scale)
    return np.asarray(normals), np.asarray(offsets)


def shape(size):
    x,y,z=np.asarray(size)/2
    section=[(-x,-z),(x,-z),(x,z),(-x+.012,z),(-x,-z+.001)]
    vertices=np.array([(a,b,c) for b in (-y,y) for a,c in section])
    faces=[]
    # Convex cap triangles and outward-oriented quadrilateral sides.
    for k in range(1,4):
        faces.extend([(0,k+1,k),(5,5+k,5+k+1)])
    for k in range(5):
        j=(k+1)%5
        faces.extend([(k,j,5+j),(k,5+j,5+k)])
    return vertices,np.asarray(faces)


def node_penetration(vertices, center, size, profile="square"):
    normals, offsets=planes(size,profile)
    distance=offsets-np.asarray(vertices-center)@normals.T
    return max(0.,float(distance.min(axis=-1).max()))


def triangle_penetration(vertices, faces, center, size, profile="square", tolerance=.001):
    if profile=="square":
        from measurements import triangles_in_box
        return triangles_in_box(vertices,faces,center,size,tolerance)
    normals,offsets=planes(size,profile)
    offsets=offsets-tolerance
    triangles=np.asarray(vertices)[faces]-center
    signed=triangles@normals.T-offsets
    candidates=triangles[~np.any(np.all(signed>0,axis=1),axis=1)]
    hits=0
    for triangle in candidates:
        polygon=list(triangle)
        for normal,offset in zip(normals,offsets):
            if not polygon:break
            clipped=[]
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                da,db=a@normal-offset,b@normal-offset
                if da<=0:clipped.append(a)
                if (da<0 and db>0) or (da>0 and db<0):
                    clipped.append(a+(b-a)*da/(da-db))
            polygon=clipped
        hits+=bool(polygon)
    return int(hits)


def sample_height(distance_from_tip,size,profile):
    if profile=="square":return 0.
    thickness=min(size[2], .001+distance_from_tip*(size[2]-.001)/.012)
    return -size[2]/2+thickness/2
