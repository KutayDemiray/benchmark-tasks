"""Estimated sewn long-sleeve shirt pattern and CPU topology checks.

Pattern coordinates u/v run across the garment and from hem to shoulder.
Back and the two overlapping fronts are sewn along shoulder/side/sleeve edges.
The front, neck, hem and cuffs remain open. No physical attachment holds a seam
to the world, hanger or robot: seams are continuous flexible surface elements.
"""

from collections import Counter, defaultdict

import numpy as np


def unique_grid(values):
    return np.unique(np.round(values, 6))


def pattern(cfg):
    """Return a connected surface mesh, resolved rivet patches and audit."""
    p = cfg['garment']
    pitch = p['body_mesh_pitch_m']
    half_span = p['half_sleeve_span_m']
    length = cfg['estimated_garment']['body_length_m']
    chest = cfg['estimated_garment']['flat_chest_width_m']/2
    shoulder = cfg['estimated_garment']['shoulder_width_m']/2
    overlap = p['placket_overlap_m']/2
    neck = p['neck_half_width_m']
    neck_bottom = length-p['neck_depth_m']
    underarm = p['underarm_height_m']
    snap_v = np.asarray(p['snap_heights_from_hem_m'])
    body_u = np.arange(-half_span, half_span+pitch/2, pitch)
    body_v = np.arange(0, length+pitch/2, pitch)
    if pitch < .01:
        # Refine the body without inserting additional 1 mm rows into the
        # existing 2 mm rivet patches or changing their attachment sampling.
        body_u = body_u[np.abs(body_u) > .012 + 1e-9]
        outside_snap_rows = np.ones(len(body_v), dtype=bool)
        for snap_height in snap_v:
            outside_snap_rows &= np.abs(body_v-snap_height) > .006 + 1e-9
        body_v = body_v[outside_snap_rows]
    us = unique_grid([*body_u,
                      *np.arange(-.012, .0121, .002), -overlap, overlap,
                      -neck, neck, -shoulder, shoulder, -chest, chest,
                      *p.get('extra_transverse_grid_m',[])])
    vs = unique_grid([*body_v, underarm, neck_bottom,
                      *(snap_v[:, None]+np.arange(-.006, .0061, .002)).ravel(),
                      *p.get('extra_height_grid_m',[])])
    assert us.min()==-half_span and us.max()==half_span and vs.max()==length
    vertices, uv, panel_ids, faces = [], [], [], []
    panel_maps = []
    panel_boundaries = []
    for panel in range(3):
        mapping, local_faces = {}, []
        depth = p.get('back_neck_depth_m',p['neck_depth_m']) if panel==0 else p['neck_depth_m']
        def vertex(i, j):
            key = (i, j)
            if key not in mapping:
                mapping[key] = len(vertices)
                u, v = float(us[i]), float(vs[j])
                sag = max(0., (abs(u)-shoulder)/(half_span-shoulder))*p['sleeve_drop_m']
                z = v-sag*min(1., v/underarm)
                vertices.append([0., u, z])
                uv.append([u, v])
                panel_ids.append(panel)
            return mapping[key]
        for j in range(len(vs)-1):
            for i in range(len(us)-1):
                u, v = (us[i]+us[i+1])/2, (vs[j]+vs[j+1])/2
                if abs(u)>chest and v<underarm:
                    continue
                cut_height = length-depth
                if p.get('neck_shape')=='round':
                    cut_height = length-depth*np.sqrt(max(0.,1-(u/neck)**2))
                if abs(u)<neck and v>cut_height:
                    continue
                if panel==1 and u < -overlap or panel==2 and u > overlap:
                    continue
                a,b,c,d = vertex(i,j),vertex(i+1,j),vertex(i,j+1),vertex(i+1,j+1)
                local_faces.extend([[a,b,c],[b,d,c]] if panel==0 else [[a,c,b],[b,c,d]])
        faces.extend(local_faces)
        counts = Counter()
        oriented = {}
        for a,b,c in local_faces:
            for edge in ((a,b),(b,c),(c,a)):
                key = tuple(sorted(edge))
                counts[key] += 1
                oriented[key] = edge
        panel_maps.append(mapping)
        panel_boundaries.append([oriented[e] for e,count in counts.items() if count==1])
    uv = np.array(uv)
    panel_ids = np.array(panel_ids)
    vertices = np.array(vertices)
    if p.get('neck_shape')=='round' and p.get('smooth_neck_boundary',False):
        # Project the stair-step cut boundary onto the estimated neck curve.
        # This removes artificial right-angle notches in the paper pattern;
        # it does not position cloth relative to the hanger or any snap.
        for panel,boundary in enumerate(panel_boundaries):
            depth = p.get('back_neck_depth_m',p['neck_depth_m']) if panel==0 else p['neck_depth_m']
            ids = set()
            for a,b in boundary:
                midpoint=(uv[a]+uv[b])/2
                if abs(midpoint[0])<neck+1e-8 and midpoint[1]>=length-depth-.001:
                    ids.update((a,b))
            for i in ids:
                vector=np.array([uv[i,0]/neck,(uv[i,1]-length)/depth])
                vector/=np.linalg.norm(vector)
                uv[i]=[vector[0]*neck,length+vector[1]*depth]
                u,v=uv[i]
                sag=max(0.,(abs(u)-shoulder)/(half_span-shoulder))*p['sleeve_drop_m']
                vertices[i,1:]=[u,v-sag*min(1.,v/underarm)]
    reverse = {index: key for mapping in panel_maps for key,index in mapping.items()}
    def is_sewn(a, b):
        u,v = (uv[a]+uv[b])/2
        return not (abs(v)<1e-8 or abs(abs(u)-half_span)<1e-8 or
                    abs(u)<neck+1e-8 and v>=neck_bottom-1e-8)
    seam_vertices = [set(),set(),set()]
    sewn_vertex_map = {}
    seam_count = 0
    for a,b in panel_boundaries[0]:
        if not is_sewn(a,b):
            continue
        panel = 1 if (uv[a,0]+uv[b,0])>0 else 2
        fa,fb = panel_maps[panel][reverse[a]],panel_maps[panel][reverse[b]]
        # Reverse each existing boundary edge to maintain a consistent surface.
        if p.get('seam_join')=='shared_vertices':
            sewn_vertex_map[fa]=a
            sewn_vertex_map[fb]=b
        else:
            faces.extend([[b,a,fa],[b,fa,fb]])
        seam_vertices[0].update((a,b))
        seam_vertices[panel].update((fa,fb))
        seam_count += 1
    from scipy.spatial import cKDTree
    depths = (p['back_depth_m'],p['outer_front_depth_m'],p['inner_front_depth_m'])
    for panel in range(3):
        ids = np.flatnonzero(panel_ids==panel)
        distance = cKDTree(uv[sorted(seam_vertices[panel])]).query(uv[ids])[0]
        blend = np.clip(distance/p['seam_transition_m'],0,1)
        if p.get('seam_profile')=='rounded':
            blend = np.sqrt(2*blend-blend*blend)
        sewn_depth = p['seam_separation_m']/2*(1 if panel==0 else -1)
        if p.get('seam_join')=='shared_vertices':
            sewn_depth=0.
        vertices[ids,0] = sewn_depth+(depths[panel]-sewn_depth)*blend
    faces = np.asarray(faces,dtype=np.int32)
    if sewn_vertex_map:
        representative=np.arange(len(vertices))
        for front,back in sewn_vertex_map.items():
            assert np.linalg.norm(vertices[front]-vertices[back])<1e-10
            representative[front]=back
        faces=representative[faces]
        used=np.unique(faces)
        compact=np.full(len(vertices),-1,dtype=np.int32)
        compact[used]=np.arange(len(used))
        faces=compact[faces]
        vertices,uv,panel_ids=vertices[used],uv[used],panel_ids[used]
    # Intrinsic rest pattern is sewn into its initial thin folded shape.
    vertices += np.asarray(p['hanging_origin_m'])
    edges = np.unique(np.sort(np.vstack((faces[:,:2], faces[:,1:],faces[:,[2,0]])),axis=1),axis=0)
    lengths = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    cross = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    area = float(np.linalg.norm(cross,axis=1).sum()/2)
    rest_positions=vertices.copy()
    closed_control=cfg.get('garment_test',{}).get('case')=='closed_target'
    def closure_displacement(u,v):
        if not closed_control:
            return np.zeros_like(np.asarray(u,dtype=float))
        # Initial-condition control only. Keep the unclosed rest metric and
        # bending shape: the fastener must retain this deformation physically.
        required_shift=depths[2]-depths[1]-.0014-.0035
        return required_shift*np.exp(-(np.asarray(u)/.055)**2-((np.asarray(v)-snap_v[0])/.035)**2)
    ids=np.flatnonzero(panel_ids==1)
    vertices[ids,0]+=closure_displacement(uv[ids,0],uv[ids,1])
    rivets = []
    for index,v in enumerate(snap_v):
        for panel,kind in ((1,'male'),(2,'socket')):
            ids = np.flatnonzero((panel_ids==panel)&(np.linalg.norm(uv-[0,v],axis=1)<.0055))
            assert len(ids)>=15, 'Unresolved fastener cloth patch'
            center=np.array(p['hanging_origin_m'])+[depths[panel],0,v]
            if panel==1:
                center[0]+=closure_displacement(0.,v)
            rivets.append(dict(name=f'{kind}{index}',panel=panel,kind=kind,index=index,
                               ids=ids,center=center))
    audit = topology_audit(rest_positions,faces)
    assert audit['connected_components']==1 and audit['boundary_loops']==3, audit
    assert audit['max_edge_valence']==2 and audit['boundary_degree_valid'] and audit['oriented'], audit
    assert audit['min_triangle_area_m2']>1e-9 and lengths.min()>.0005, audit
    audit.update(area_m2=area,fabric_mass_kg=area*cfg['fabric']['areal_density_kg_m2'],
                 sewn_segments=seam_count,rivet_patch_vertices=[len(r['ids']) for r in rivets])
    return dict(initial=vertices,rest_positions=rest_positions,faces=faces,edges=edges,rest_lengths=lengths,
                uv=uv,panel_ids=panel_ids,rivets=rivets,audit=audit)


def topology_audit(vertices,faces):
    """Check manifold boundaries, winding and connectivity without simulation."""
    incidence = defaultdict(list)
    neighbors = defaultdict(set)
    for a,b,c in faces:
        for a,b in ((a,b),(b,c),(c,a)):
            incidence[tuple(sorted((a,b)))].append((a,b))
            neighbors[a].add(b)
            neighbors[b].add(a)
    boundary = defaultdict(set)
    for key,items in incidence.items():
        if len(items)==1:
            a,b = key
            boundary[a].add(b)
            boundary[b].add(a)
    def components(graph):
        remaining = set(graph)
        count = 0
        while remaining:
            pending = [remaining.pop()]
            while pending:
                next_nodes = graph[pending.pop()]&remaining
                pending.extend(next_nodes)
                remaining.difference_update(next_nodes)
            count += 1
        return count
    cross = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    return dict(vertices=len(vertices),triangles=len(faces),edges=len(incidence),
                connected_components=components(neighbors),boundary_loops=components(boundary),
                boundary_degree_valid=all(len(n)==2 for n in boundary.values()),
                max_edge_valence=max(map(len,incidence.values())),
                oriented=all(len(e)!=2 or e[0]==tuple(reversed(e[1])) for e in incidence.values()),
                referenced_vertices=len(neighbors),min_triangle_area_m2=float(np.linalg.norm(cross,axis=1).min()/2))


if __name__=='__main__':
    import json
    from pathlib import Path
    data = pattern(json.loads((Path(__file__).parent/'parameters.json').read_text()))
    print(json.dumps(data['audit'],indent=2))
