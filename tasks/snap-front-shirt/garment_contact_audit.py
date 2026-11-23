"""CPU geometric checks for cloth passing through the physical hanger bars."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

import numpy as np

from robot_geometry import rotation


def hook_polyline(hanger):
    """Return the authored hook cylinder centreline, including its stem link."""
    angles=np.linspace(np.deg2rad(217),np.deg2rad(-25),31)
    arc=np.asarray(hanger['hook_center_local_m'])+hanger['hook_radius_m']*np.column_stack(
        (np.cos(angles),np.zeros(len(angles)),np.sin(angles)))
    return np.vstack(([0,0,.04],arc))


def initial_hook_rail_gap(hanger):
    """Conservatively bound hook/rail clearance for an identity reset pose."""
    points=hook_polyline(hanger)+np.asarray(hanger['origin_m'])
    # Hook is in the middle of a long Y-axis rail. Capsule bounds enclose
    # the finite hook cylinders; a positive result proves separation.
    a=points[:-1][:,[0,2]]
    d=(points[1:]-points[:-1])[:,[0,2]]
    w=np.asarray(hanger['rail_center_m'])[[0,2]]-a
    f=np.clip(np.einsum('ij,ij->i',w,d)/np.einsum('ij,ij->i',d,d),0,1)
    return float(np.linalg.norm(w-f[:,None]*d,axis=1).min()
                 -hanger['rail_radius_m']-hanger['hook_wire_radius_m'])


def hook_rail_penetration(pose,hanger):
    """Find penetration witnesses on actual hook cylinder surfaces.

    Sampling can prove an overlap but cannot prove its absence. It covers the
    long rail shaft, not its ends. Returned depth is a lower bound on maximum
    penetration, distinct from the conservative reset clearance bound.
    """
    points=hook_polyline(hanger)
    samples=[]
    angles=np.arange(64)*2*np.pi/64
    for a,b in zip(points[:-1],points[1:]):
        axis=(b-a)/np.linalg.norm(b-a)
        u=np.cross(axis,[0.,1.,0.])
        v=np.cross(axis,u)
        ring=hanger['hook_wire_radius_m']*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
        for f in (0.,.5,1.):
            samples.append(a+f*(b-a)+ring)
    world=np.vstack(samples)@rotation(pose[3:]).T+np.asarray(pose[:3])
    rel=world-np.asarray(hanger['rail_center_m'])
    inside_shaft=np.abs(rel[:,1])<hanger['rail_length_m']/2
    depth=np.where(inside_shaft,hanger['rail_radius_m']-np.linalg.norm(rel[:,[0,2]],axis=1),0.)
    index=int(np.argmax(depth))
    return dict(penetration_lower_bound_m=float(max(0.,depth[index])),
                witness_world_m=world[index].tolist(),method='sampled_hook_cylinder_surface_inside_rail_shaft')


def segment_triangle_distances(tri,a,b):
    """Return exact distances from a segment to triangles, excluding crossings.

    Segment/triangle intersections must separately be assigned distance zero.
    Endpoint/face and segment/edge feature pairs cover the other minima.
    """
    from trimesh.triangles import closest_point
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    d=b-a
    aa=float(d@d)
    pa=closest_point(tri,np.broadcast_to(a,(len(tri),3)).copy())
    pb=closest_point(tri,np.broadcast_to(b,(len(tri),3)).copy())
    best=np.minimum(np.sum((pa-a)**2,axis=1),np.sum((pb-b)**2,axis=1))
    for i,j in ((0,1),(1,2),(2,0)):
        q=tri[:,i]
        e=tri[:,j]-q
        w=a-q
        bb=e@d
        cc=np.einsum('ij,ij->i',e,e)
        dd=w@d
        ee=np.einsum('ij,ij->i',w,e)
        den=aa*cc-bb*bb
        ss=np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-20)
        tt=np.divide(aa*ee-bb*dd,den,out=np.zeros_like(den),where=den>1e-20)
        valid=(den>1e-20)&(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1)
        delta=w+ss[:,None]*d-tt[:,None]*e
        best=np.minimum(best,np.where(valid,np.einsum('ij,ij->i',delta,delta),np.inf))
        for end in (q,q+e):
            f=np.clip((end-a)@d/aa,0,1)
            delta=end-(a+f[:,None]*d)
            best=np.minimum(best,np.einsum('ij,ij->i',delta,delta))
    return np.sqrt(np.maximum(best,0.))


def initial_capsule_clearances(nodes,faces,pose,hanger):
    """Check complete triangles against the three capsule body bars at reset."""
    local=(np.asarray(nodes,dtype=float)-np.asarray(pose[:3]))@rotation(pose[3:])
    tri=local[np.asarray(faces)]
    half,drop=hanger['width_m']/2,hanger['arm_drop_m']
    bars=[([0,0,0],[0,half,-drop]),([0,0,0],[0,-half,-drop]),([0,-half,-drop],[0,half,-drop])]
    crossings=bar_contacts(nodes,faces,pose,hanger)
    result=[]
    for index,(a,b) in enumerate(bars):
        distance=segment_triangle_distances(tri,a,b)
        distance[crossings[index]['crossed_triangles']]=0.
        result.append(float(distance.min()-hanger['body_radius_m']))
    return result


def bar_contacts(nodes,faces,pose,hanger):
    """Report centreline crossings and penetration of authored hanger bars."""
    local=(np.asarray(nodes,dtype=np.float64)-np.asarray(pose[:3]))@rotation(pose[3:])
    tri=local[faces]
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    half,drop=hanger['width_m']/2,hanger['arm_drop_m']
    bars=[([0,0,0],[0,half,-drop]),([0,0,0],[0,-half,-drop]),([0,-half,-drop],[0,half,-drop])]
    wooden=hanger.get('body_style')=='wooden'
    if wooden:
        from hanger_shapes import wooden_beams,triangle_box_intersections
        beams=wooden_beams(hanger)
        end=half-hanger['crossbar_radius_m']
        bars=[(beam['a'],beam['b']) for beam in beams]+[([0,-end,-drop],[0,end,-drop])]
    result=[]
    for index,(a,b) in enumerate(bars):
        a,b=np.asarray(a),np.asarray(b)
        direction=b-a
        h=np.cross(direction,e2)
        det=np.einsum('ij,ij->i',e1,h)
        valid=np.abs(det)>1e-12
        inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
        s=a-tri[:,0]
        u=inv*np.einsum('ij,ij->i',s,h)
        q=np.cross(s,e1)
        v=inv*(q@direction)
        t=inv*np.einsum('ij,ij->i',e2,q)
        crossed=valid&(u>=0)&(v>=0)&(u+v<=1)&(t>=0)&(t<=1)
        relative=local-a
        axial=relative@direction/(direction@direction)
        radial=np.linalg.norm(relative-axial[:,None]*direction,axis=1)
        radius=hanger['crossbar_radius_m'] if wooden else hanger['body_radius_m']
        depth=np.where((axial>0)&(axial<1),radius-radial,0.)
        extra={}
        if wooden and index<2:
            beam=beams[index]
            box_nodes=(local-beam['center'])@beam['rotation']
            half_size=beam['size']/2
            depth=(half_size-np.abs(box_nodes)).min(axis=1)
            extra=dict(deeply_intersecting_triangles=np.flatnonzero(
                triangle_box_intersections(box_nodes[faces],half_size-.001)).tolist(),
                geometry='physical_rectangular_wooden_shoulder')
        result.append(dict(crossed_triangles=np.flatnonzero(crossed).tolist(),
                      maximum_vertex_penetration_m=float(max(0.,depth.max())),
                      deeply_penetrating_vertices=np.flatnonzero(depth>.001).tolist(),**extra))
    return result


def inspect(episode,physics_substeps=False):
    """Audit initial and recorded hanger contact states; do not edit outcomes."""
    setup=json.loads((episode/'setup.json').read_text())
    topology=json.loads((episode/'topology.json').read_text())
    if physics_substeps:
        physical=[json.loads(x) for x in (episode/'physics.jsonl').read_text().splitlines()]
        frames=np.load(episode/'contact-substeps.npy',mmap_mode='r')
        count=min(len(physical),len(frames))
        frames=frames[:count]
        hz=setup['parameters']['simulation']['physics_hz']
        trace=[dict(time_s=r['physics_step']/hz,physics=r) for r in physical[:count]]
    else:
        trace=[json.loads(x) for x in (episode/'trace.jsonl').read_text().splitlines()]
        frames=np.load(episode/'control-steps.npz')['cloth_nodes']
    faces=np.asarray(topology['cloth']['faces'])
    cfg=setup['parameters']['hanger']
    initial=bar_contacts(topology['cloth']['initial'],faces,topology['initial_hardware']['hanger'],cfg)
    records=[]
    for row,nodes in zip(trace,frames,strict=True):
        values=bar_contacts(nodes,faces,row['physics']['hardware']['hanger'],cfg)
        hook=hook_rail_penetration(row['physics']['hardware']['hanger'],cfg)
        records.append(dict(time_s=row['time_s'],bars=values,hook_rail=hook))
    return dict(initial=initial,frames=records,sampling=('native_physics_rate' if physics_substeps else '10_Hz'),
                centreline_crossing_frames=sum(any(b['crossed_triangles'] for b in r['bars']) for r in records),
                max_vertex_penetration_m=max((b['maximum_vertex_penetration_m'] for r in records for b in r['bars']),default=0.),
                max_hook_rail_penetration_lower_bound_m=max((r['hook_rail']['penetration_lower_bound_m'] for r in records),default=0.),
                limitation='Recorded geometry only; absence of crossings does not exclude between-sample penetration. Hook/rail sampling proves overlaps only; garment self-contact and bar ends need separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--physics-substeps',action='store_true')
    args=parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    result=inspect(args.episode,args.physics_substeps)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('centreline_crossing_frames','max_vertex_penetration_m')}))
