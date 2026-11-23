"""Audit passive cloth/frame geometry without assigning any benchmark score."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from audit_mechanism import capsule_proxy
from mesh import surface_edges
from passive_contact import native_physics_errors
from robot_contacts import penetrating_gripper_shapes
from self_contact import crossings, separating_translations
from triangle_broadphase import TriangleBroadphase


def audit(run, stride):
    """Check every selected native state using the existing contact tolerances."""
    request=json.loads((run/'request.json').read_text())
    result=json.loads((run/'result.json').read_text())
    assert request['experiment']=='passive_contact' and result['status']=='completed'
    assert json.loads((run/'process.json').read_text())['returncode']==0
    assert not native_physics_errors((run/'kit.log').read_text())
    proof=json.loads((run/'passive-initialization.json').read_text())
    assert not proof['robots_present'] and not proof['controllers_present']
    scene_path=run.parent/proof['source_full_run']/'scene-audit.json'
    scene=json.loads(scene_path.read_text())
    assert not scene['cloth_attachments']
    names=json.loads((run/'frame-names.json').read_text())['body_names']
    frame=json.loads((run/'frame-geometry.json').read_text())
    shapes=[]
    for shape in frame['colliders']:
        if not shape['enabled']:continue
        vertices=capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
        shapes.append(dict(body=shape['body'],name='Frame/'+shape['body']+'/'+shape['name'],points=vertices,planes=ConvexHull(vertices).equations))
    vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))*np.array(scene['table_size_m'])/2
    shapes.append(dict(body='Table',name='Table',points=vertices,planes=ConvexHull(vertices).equations))
    rigid=dict(body_names=[*names,'Table'],colliders=shapes)
    table_pose=np.r_[scene['table_center_m'],[0.,0.,0.,1.]]
    with np.load(run/'geometry.npz') as geometry:
        faces,rest=geometry['faces'].copy(),geometry['rest'].copy()
    edges=surface_edges(faces);rest_length=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    broadphase=TriangleBroadphase(rest,faces)
    peak_strain=peak_crossing=0.;violations=[];checked=total=0;hashes={}
    for path in sorted((run/'physics-chunks').glob('*.npz')):
        hashes[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:
            for local,points in enumerate(chunk['positions']):
                step=total;total+=1
                if step%stride and step!=result['physics_steps']-1:continue
                poses=chunk['frame_body_poses'][local]
                assert np.isfinite(points).all() and np.isfinite(poses).all()
                peak_strain=max(peak_strain,float(abs(np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)/rest_length-1).max()))
                hits=penetrating_gripper_shapes(points,faces,np.r_[poses,table_pose[None]],rigid)
                pairs,_=crossings(points,faces,broadphase)
                if len(pairs):
                    depth=float(np.linalg.norm(separating_translations(points,faces,pairs),axis=1).max())
                    peak_crossing=max(peak_crossing,depth)
                    if depth>.001:hits.append('Cloth/nonadjacent_self_crossing')
                if hits:violations.append(dict(physics_step=step,colliders=hits))
                checked+=1
        print(json.dumps(dict(chunk=path.name,checked=checked,strain=peak_strain,violating_states=len(violations))),flush=True)
    assert total==result['physics_steps']
    for name in ('request.json','result.json','geometry.npz','frame-names.json','frame-geometry.json','passive-initialization.json'):
        hashes[name]=hashlib.sha256((run/name).read_bytes()).hexdigest()
    report=dict(diagnostic_only=True,benchmark_eligible=False,task_score=None,stride=stride,
        complete_native_coverage=stride==1 and checked==total,checked_states=checked,total_states=total,
        max_abs_edge_strain=peak_strain,self_crossing_penetration_upper_bound_m=peak_crossing,
        violations=violations,selected_states_passed=peak_strain<=.2 and not violations,
        source_sha256=hashes,table_source_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Conservative capsule outer hulls; rigid/rigid contacts are not audited here.',
            'A stride above one is only a screen and cannot establish complete contact validity.'])
    (run/f'passive-geometry-stride-{stride}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','violations')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--stride',type=int,default=1)
    args=parser.parse_args()
    assert args.stride>=1 and args.run.resolve().is_relative_to(OUTPUT_ROOT)
    audit(args.run,args.stride)
