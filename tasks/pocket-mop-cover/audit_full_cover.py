"""Audit every recorded cloth substep against the complete native robot scene."""

from task_paths import OUTPUT_ROOT

import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull

from audit_mechanism import capsule_proxy
from mesh import surface_edges
from robot_contacts import penetrating_gripper_shapes
from self_contact import crossings,separating_translations
from triangle_broadphase import TriangleBroadphase


def recorded_chunks(run,follow):
    """Yield completed chunks, waiting for an active recorder when requested."""
    consumed=0
    while True:
        finished=(run/'result.json').exists()
        paths=sorted((run/'physics-chunks').glob('[0-9][0-9][0-9][0-9][0-9].npz'))
        # Older recorders write directly to the final filename. The next chunk
        # or the final result proves that the previous file has finished writing.
        available=len(paths) if finished or not follow else max(0,len(paths)-1)
        while consumed<available:
            yield paths[consumed]
            consumed+=1
        if finished or not follow:return
        time.sleep(2.)


def audit(run,follow=False,chunk_range=None,diagnostic_error_run=False):
    """Write a complete material/contact certificate bound to the recorded files."""
    request=json.loads((run/'request.json').read_text())
    assert request.get('robot_cover')
    if not follow:
        status=json.loads((run/'result.json').read_text())['status']
        assert status=='completed' or (diagnostic_error_run and status=='error')
    scene=json.loads((run/'scene-audit.json').read_text())
    frame=json.loads((run/'frame-geometry.json').read_text())
    names=json.loads((run/'frame-names.json').read_text())
    robot=json.loads((run/'robot-cooked-colliders.json').read_text())
    proof=robot.pop('_provenance')
    assert proof['source_run']==run.name and not proof['scene_modified']
    for name,expected in proof['source_sha256'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected
    geometry=np.load(run/'geometry.npz')
    faces=geometry['faces'];edges=surface_edges(faces)
    broadphase=TriangleBroadphase(geometry['rest'],faces)
    rest_lengths=np.linalg.norm(geometry['rest'][edges[:,1]]-geometry['rest'][edges[:,0]],axis=1)
    frame_shapes=[]
    for shape in frame['colliders']:
        if not shape['enabled']:continue
        points=capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
        frame_shapes.append(dict(body=shape['body'],name='Frame/'+shape['body']+'/'+shape['name'],
            points=points,planes=ConvexHull(points).equations))
    points=np.array(list(itertools.product((-1.,1.),repeat=3)))*np.array(scene['table_size_m'])/2
    frame_shapes.append(dict(body='Table',name='Table',points=points,planes=ConvexHull(points).equations))
    rigid_geometry=dict(body_names=[*names['body_names'],'Table'],colliders=frame_shapes)
    table_pose=np.r_[scene['table_center_m'],[0.,0.,0.,1.]]
    peak_strain=0.;peak_crossing=0.
    native_speeds=deque(maxlen=2*request['physics_hz'])
    difference_speeds=deque(maxlen=2*request['physics_hz'])
    finite=True;violations=[];steps=0
    first_difference_speed=None
    previous=geometry['initial']
    source_hashes={}
    step_offset=0
    output_stem='full-cloth-audit'
    paths=recorded_chunks(run,follow)
    all_paths=None
    if chunk_range is not None:
        assert not follow
        all_paths=list(paths)
        start,stop=chunk_range
        assert 0<=start<stop<=len(all_paths)
        assert [int(path.stem) for path in all_paths]==list(range(len(all_paths)))
        paths=all_paths[start:stop]
        step_offset=start*request['physics_hz']
        if start:
            source_hashes[str(all_paths[start-1].relative_to(run))]=hashlib.sha256(all_paths[start-1].read_bytes()).hexdigest()
            with np.load(all_paths[start-1]) as previous_chunk:
                previous=previous_chunk['positions'][-1]
        output_stem+=f'-part-{start:05d}-{stop:05d}'
    checked_chunks=0
    for path in paths:
        source_hashes[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:
            x=chunk['positions'];v=chunk['velocities'];poses=chunk['frame_body_poses']
            checked_chunks+=1
            if all_paths is not None and path!=all_paths[-1]:assert len(x)==request['physics_hz']
            robot_poses={side:chunk[side+'_body_poses'] for side in ('left','right')}
            finite=finite and bool(np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(poses).all()
                and all(np.isfinite(value).all() for value in robot_poses.values()))
            assert len(x)==len(v)==len(poses)
            if not finite:
                steps+=len(x)
                continue
            for index,points in enumerate(x):
                lengths=np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)
                peak_strain=max(peak_strain,float(abs(lengths/rest_lengths-1).max()))
                hits=penetrating_gripper_shapes(points,faces,np.r_[poses[index],table_pose[None]],rigid_geometry)
                for side in ('left','right'):
                    hits.extend(side+name for name in penetrating_gripper_shapes(points,faces,
                        robot_poses[side][index],robot[side]))
                pairs,_=crossings(points,faces,broadphase)
                if len(pairs):
                    depth=float(np.linalg.norm(separating_translations(points,faces,pairs),axis=1).max())
                    peak_crossing=max(peak_crossing,depth)
                    if depth>.001:hits.append('Cloth/nonadjacent_self_crossing')
                if hits:violations.append(dict(physics_step=step_offset+steps,colliders=hits))
                native_speeds.append(float(np.linalg.norm(v[index],axis=1).max()))
                difference_speeds.append(float(np.linalg.norm(points-previous,axis=1).max()*request['physics_hz']))
                if first_difference_speed is None:first_difference_speed=difference_speeds[-1]
                previous=points
                steps+=1
                if steps%1000==0:
                    heartbeat=dict(physics_steps=steps,control_limit_steps=round(request['seconds']*20),
                        following_active_run=follow and not (run/'result.json').exists(),
                        violating_steps=len(violations),max_abs_edge_strain=peak_strain)
                    (run/(output_stem+'-heartbeat.json')).write_text(json.dumps(heartbeat))
                    print(json.dumps(heartbeat),flush=True)
    result=json.loads((run/'result.json').read_text())
    complete=chunk_range is None and result['status']=='completed' and steps==result['physics_steps']==result['control_steps']*(request['physics_hz']//20)
    for name in ('geometry.npz','cover-mesh.json','frame-geometry.json','frame-names.json','robot-cooked-colliders.json',
                 'scene-audit.json','request.json','manifest.json','scene.usda','steps.jsonl','result.json'):
        source_hashes[name]=hashlib.sha256((run/name).read_bytes()).hexdigest()
    report=dict(scope='Every native material substep; frame, table and all cooked robot colliders.',
        complete=complete,physics_steps=steps,finite=finite,max_abs_edge_strain=peak_strain,
        recording_status=result['status'],
        diagnostic_only=result['status']!='completed' or bool(request.get('release_probe_run')),
        benchmark_eligible=result['status']=='completed' and not bool(request.get('release_probe_run')),
        chunk_range=chunk_range,physics_step_offset=step_offset,checked_chunks=checked_chunks,
        self_crossing_penetration_upper_bound_m=peak_crossing,violations=violations,
        max_seam_gap_m=0.,seam_evidence='All three sewn edges share fixed native material vertices; no tear model or runtime topology writes.',
        no_attachment=not scene['cloth_attachments'],native_final_two_second_speed_m_s=max(native_speeds,default=0.),
        difference_final_two_second_speed_m_s=max(difference_speeds,default=0.),source_sha256=source_hashes,
        first_difference_speed_m_s=first_difference_speed,
        numerical_checks_passed=complete and finite and peak_strain<=.2 and not violations and not scene['cloth_attachments'],
        recorded_numerical_checks_passed=finite and peak_strain<=.2 and not violations and not scene['cloth_attachments'],
        crossing_search='Refitted triangle bounding boxes; original crossing narrow phase and tolerances.',
        limitations=['Frame capsules use conservative outer convex bounds.',
            'A complete separate rigid-body audit is required before outcome scoring.'])
    (run/(output_stem+'.json')).write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--diagnostic-error-run',action='store_true',help='Inspect a stopped recording without qualifying it for a score')
    args=parser.parse_args()
    assert args.run.resolve().parent==OUTPUT_ROOT
    result=audit(args.run,args.follow,diagnostic_error_run=args.diagnostic_error_run)
    result['violations']=len(result['violations']);result.pop('source_sha256')
    print(json.dumps(result,indent=2))
