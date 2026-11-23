"""Audit all native passive frame states with the existing rigid contact tolerances."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_mechanism import capsule_proxy,features,penetration
from passive_contact import native_physics_errors


def audit(run):
    result=json.loads((run/'result.json').read_text())
    request=json.loads((run/'request.json').read_text())
    assert request['experiment']=='passive_contact' and result['status']=='completed'
    assert request['passive_mode']=='articulated' and not request.get('passive_free_space')
    assert not native_physics_errors((run/'kit.log').read_text())
    names=json.loads((run/'frame-names.json').read_text())['body_names']
    metadata=json.loads((run/'frame-geometry.json').read_text())
    proof=json.loads((run/'passive-initialization.json').read_text())
    scene_path=run.parent/proof['source_full_run']/'scene-audit.json'
    scene=json.loads(scene_path.read_text());table_top=scene['table_center_m'][2]+scene['table_size_m'][2]/2
    samples=[];hashes={}
    for path in sorted((run/'physics-chunks').glob('*.npz')):
        hashes[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:samples.append(chunk['frame_body_poses'])
    bodies=np.concatenate(samples);assert len(bodies)==result['physics_steps'] and np.isfinite(bodies).all()
    floor=0.;shapes=[];excess=0.
    for shape in metadata['colliders']:
        if not shape['enabled']:continue
        pose=bodies[:,names.index(shape['body'])]
        rotations=Rotation.from_quat(pose[:,3:]).as_matrix()
        if 'capsule' in shape:
            vertices,extra=capsule_proxy(shape['capsule']);excess=max(excess,extra)
        else:vertices=np.asarray(shape['points'])
        normals,edges=features(vertices)
        world=np.einsum('tij,pj->tpi',rotations,vertices)+pose[:,None,:3]
        floor=max(floor,float(table_top-world[:,:,2].min()))
        shapes.append((shape,(pose[:,:3],rotations,vertices,normals,edges)))
    adjacent={frozenset(pair) for pair in (('Bridge','Left'),('Bridge','Right'),('Bridge','Connector'),('Connector','Handle'),('Rocker','Clip'))}
    overlaps=[];maximum=0.
    for (sa,a),(sb,b) in itertools.combinations(shapes,2):
        pair=frozenset((sa['body'],sb['body']))
        if len(pair)==1 or pair in adjacent:continue
        depths=penetration(a,b,upper_bound_tolerance=.001)
        peak=float(depths.max());maximum=max(maximum,peak)
        if peak>1e-6:
            overlaps.append(dict(first=sa['body']+'/'+sa['name'],second=sb['body']+'/'+sb['name'],
                max_penetration_bound_m=peak,violating_steps=np.flatnonzero(depths>.001).tolist()))
    for name in ('request.json','result.json','frame-names.json','frame-geometry.json','passive-initialization.json'):
        hashes[name]=hashlib.sha256((run/name).read_bytes()).hexdigest()
    report=dict(diagnostic_only=True,benchmark_eligible=False,complete_native_coverage=True,
        physics_steps=len(bodies),floor_penetration_bound_m=floor,
        nonadjacent_convex_penetration_bound_m=maximum,overlapping_pairs=overlaps,
        rigid_checks_passed=maximum<=.001 and floor<=.001,
        adjacent_hinge_pairs_excluded=[sorted(pair) for pair in sorted(adjacent,key=lambda p:sorted(p))],
        capsule_outer_bound_radial_excess_m=excess,source_sha256=hashes,
        table_source_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Uses the same hinge-bearing exclusions as the existing isolated mechanism audit.',
            'Capsule hulls and early SAT rejection provide conservative upper bounds.',
            'Cloth contact and strain require the separate all-state geometry audit.'])
    (run/'passive-rigid-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','overlapping_pairs')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();assert args.run.resolve().is_relative_to(OUTPUT_ROOT)
    audit(args.run)
