"""Independently audit one sewn pocket under the physical folded-frame rig."""

from task_paths import OUTPUT_ROOT
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from audit_mechanism import capsule_proxy
from measurements import closed_pocket_faces, pocket_seating
from mesh import surface_edges
from robot_contacts import penetrating_gripper_shapes
from self_contact import crossings, separating_translations


def audit(run):
    req=json.loads((run/'request.json').read_text())
    result=json.loads((run/'result.json').read_text())
    assert (req.get('with_pocket') or req.get('with_cover')) and result['status']=='completed'
    two_pockets=req.get('with_cover',False)
    meta=json.loads((run/'mechanism.json').read_text())
    g=np.load(run/'geometry.npz')
    cloth=np.load(run/'cloth-states.npz')
    rigid=np.load(run/'states.npz')
    rows=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    x,v=cloth['positions'],cloth['velocities']
    hz=req['physics_hz']
    assert len(x)==len(v)==len(rows)==len(rigid['body_poses'])
    body_names=rigid['body_names'].tolist()
    colliders=[]
    for c in meta['colliders']:
        if not c['enabled']:
            continue
        points=capsule_proxy(c['capsule'])[0] if 'capsule' in c else np.asarray(c['points'])
        colliders.append(dict(body=c['body'],name=c['body']+'/'+c['name'],points=points,
                              planes=ConvexHull(points).equations))
    geometry=dict(body_names=body_names,colliders=colliders)
    pusher_geometry=None
    if 'pedal_pusher' in meta:
        shapes=[]
        for body,size in [('Finger',[.02,.025,.01]),('Base',[.02,.02,.03])]:
            half=np.asarray(size)/2
            points=np.array([[a,b,c] for a in [-half[0],half[0]] for b in [-half[1],half[1]] for c in [-half[2],half[2]]])
            shapes.append(dict(body=body,name='PedalPusher/'+body,points=points,planes=ConvexHull(points).equations))
        pusher_geometry=dict(body_names=rigid['pusher_body_names'].tolist(),colliders=shapes)
    markers=json.loads((run/'mesh.json').read_text())['markers']
    closed=closed_pocket_faces(g['rest'],g['faces'],markers) if not two_pockets else None
    if two_pockets:
        pocket_geometry=json.loads((run/'cover-mesh.json').read_text())['pockets']
        closed=np.asarray(pocket_geometry[0]['closed_faces'])
        left_faces=np.asarray(pocket_geometry[0]['faces'])
    else:
        left_faces=g['faces']
    edges=surface_edges(g['faces'])
    rest_length=np.linalg.norm(g['rest'][edges[:,1]]-g['rest'][edges[:,0]],axis=1)
    peak_strain=0.
    for start in range(0,len(x),64):
        lengths=np.linalg.norm(x[start:start+64,edges[:,1]]-x[start:start+64,edges[:,0]],axis=2)
        peak_strain=max(peak_strain,float(abs(lengths/rest_length-1).max()))
    hits=[];hit_names=set();cross_steps=[];max_span=0.;samples=[]
    max_crossing_depth=0.
    for i,points in enumerate(x):
        penetrated=penetrating_gripper_shapes(points,g['faces'],rigid['body_poses'][i],geometry)
        if pusher_geometry is not None:
            penetrated+=penetrating_gripper_shapes(points,g['faces'],rigid['pusher_poses'][i],pusher_geometry)
        if penetrated:
            hits.append(i);hit_names.update(penetrated)
        pairs,spans=crossings(points,g['faces'])
        if len(pairs):
            max_span=max(max_span,float(spans.max()))
            depth=float(np.linalg.norm(separating_translations(points,g['faces'],pairs),axis=1).max())
            max_crossing_depth=max(max_crossing_depth,depth)
            if depth>.001:cross_steps.append(i)
        if i%(hz//20)==0 or i==len(x)-1:
            pose=rigid['body_poses'][i,body_names.index('Left')]
            rotation=Rotation.from_quat(pose[3:]).as_matrix()
            centre=pose[:3]+rotation@np.array([-.142,0.,0.])
            seat=pocket_seating(points,closed,centre,[.1,.1,.006],markers=markers,faces=left_faces,profile='beveled',rotation=rotation)
            sample=dict(step=i,time_s=rows[i]['time_s'],phase=rows[i]['phase'],**seat)
            if two_pockets:
                right_pose=rigid['body_poses'][i,body_names.index('Right')]
                right_rotation=Rotation.from_quat(right_pose[3:]).as_matrix()
                right_centre=right_pose[:3]+right_rotation@np.array([.142,0.,0.])
                other=pocket_geometry[1]
                right=pocket_seating(points,np.asarray(other['closed_faces']),right_centre,[.1,.1,.006],
                    markers=other['markers'],faces=np.asarray(other['faces']),profile='beveled',
                    rotation=right_rotation@np.diag([-1.,-1.,1.]))
                sample.update(right_pocket=right,both_seated=seat['seated_samples'] and right['seated_samples'])
            samples.append(sample)
        if i%1000==0:
            progress=dict(step=i,total=len(x),penetrating_steps=len(hits),self_crossing_violations=len(cross_steps))
            (run/'cloth-audit-heartbeat.json').write_text(json.dumps(progress))
            print(json.dumps(progress),flush=True)
    longest=current=0
    for s in samples[:-1]:
        current=current+1 if s['seated_samples'] else 0
        longest=max(longest,current)
    speed=float(np.linalg.norm(v[-2*hz:],axis=2).max())
    difference=float(np.linalg.norm(np.diff(x[-2*hz:],axis=0),axis=2).max()*hz)
    floor=max(0.,-float(x[:,:,2].min()))
    finite=bool(np.isfinite(x).all() and np.isfinite(v).all())
    numerical=finite and peak_strain<=.2 and floor<=.001 and not hits and not cross_steps
    report=dict(scope='Two-pocket folded-frame laboratory retention; no robot or task score.' if two_pockets else 'Single-pocket folded-frame laboratory cycle; no robot, second pocket or task score.',
        diagnostic_only=True,full_task_success=False,physics_steps=len(x),samples=samples,
        max_abs_edge_strain=peak_strain,max_ground_penetration_m=floor,
        penetrating_steps=hits,penetrated_colliders=sorted(hit_names),self_crossing_violation_steps=cross_steps,
        self_crossing_normal_span_estimate_m=max_span,seated_contiguous_seconds=longest/20,
        self_crossing_penetration_upper_bound_m=max_crossing_depth,
        final_seating=samples[-1],native_final_two_second_speed_m_s=speed,
        difference_final_two_second_speed_m_s=difference,numerical_checks_passed=numerical,
        release_settling_passed=speed<=.03 and difference<=.03,
        complete_withdrawal=samples[-1]['inside_fraction']==0.,
        no_attachment=not json.loads((run/'scene-audit.json').read_text())['cloth_attachments'],
        mechanism_contact_audit_required=True)
    report['single_pocket_cycle_checks_passed']=bool(numerical and report['no_attachment'] and longest>=40
        and report['release_settling_passed'] and report['complete_withdrawal'])
    if two_pockets:
        joint_names=rigid['joint_names'].tolist()
        wing=np.rad2deg(rigid['joint_positions'][:,[joint_names.index('HingeLeft'),joint_names.index('HingeRight')]])
        retained=[j for j,row in enumerate(rows) if row['phase']=='retention' and row['time_s']>=10.]
        max_wing=float(abs(wing[retained]).max()) if retained else 999.
        min_clearance=float(x[retained,:,2].min()) if retained else -999.
        final_samples=[sample for sample in samples if sample['time_s']>rows[-1]['time_s']-2.]
        both_held=bool(final_samples and all(sample['both_seated'] for sample in final_samples))
        report.update(two_pockets=True,max_retained_wing_angle_deg=max_wing,
            min_retained_all_cloth_clearance_m=min_clearance,both_pockets_final_two_seconds=both_held,
            two_pocket_retention_checks_passed=bool(numerical and report['no_attachment'] and both_held
                and max_wing<=3. and min_clearance>=.1 and report['release_settling_passed']),
            single_pocket_cycle_checks_passed=False)
    report['auditor_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['audit_folded_pocket.py','measurements.py','robot_contacts.py','self_contact.py']}
    (run/'folded-pocket-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args()
    assert args.run.resolve().parent==OUTPUT_ROOT
    result=audit(args.run)
    for key in ['samples','penetrating_steps','self_crossing_violation_steps']:
        result[key]=len(result[key])
    print(json.dumps(result,indent=2))
