"""Audit every native rigid contact without assuming a controller or endpoint."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import importlib.util
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import sys

import numpy as np


HELPERS = ('audit_frame_robot.py', 'audit_mechanism.py', 'sat_accelerated.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def archived_helpers(run):
    """Load the verified archived helpers for this audit and restore module state."""
    manifest = json.loads((run / 'manifest.json').read_text())['files']
    previous = {name.removesuffix('.py'): sys.modules.get(name.removesuffix('.py')) for name in HELPERS}
    sys.path.insert(0, str(run / 'source'))
    try:
        # Load dependencies first so every imported algorithm is the recorded one.
        for name in reversed(HELPERS):
            path = run / 'source' / name
            assert digest(path) == manifest[name], 'Contact helper changed: ' + name
            key = name.removesuffix('.py')
            spec = importlib.util.spec_from_file_location(key, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        yield sys.modules['audit_frame_robot']
    finally:
        sys.path.pop(0)
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def preserve_assessor(run):
    """Archive this post-recording assessor without changing the launch archive."""
    source = Path(__file__).resolve()
    archive = run / 'assessment-source' / source.name
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        assert digest(archive) == digest(source), 'A different assessor is already archived'
    else:
        archive.write_bytes(source.read_bytes())
    configuration = source.with_name('task_paths.py')
    archived_configuration = archive.with_name('task_paths.py')
    if archived_configuration.exists():
        assert digest(archived_configuration) == digest(configuration), 'Archived path configuration changed'
    else:
        archived_configuration.write_bytes(configuration.read_bytes())
    return archive


def audit(run, diagnostic_error_run=False):
    """Measure all contacts; leave installation and retention to the outcome scorer."""
    with archived_helpers(run) as helpers:
        return _audit(run, helpers, diagnostic_error_run)


def _audit(run, helpers, diagnostic_error_run=False):
    assert not diagnostic_error_run, 'This assessor requires a completed episode'
    prepare, compare = helpers.prepare, helpers.compare
    assessor = preserve_assessor(run)
    assert not (run/"invalid-setup.json").exists(), "Known invalid setup cannot qualify"
    from yam_arena import station
    request=json.loads((run/'request.json').read_text())
    if request.get("table_origin",False):
        checked=json.loads((run/"station-pose-check.json").read_text())
        assert {"floor","table","LeftArm","RightArm","TopCam"}.issubset(checked)
        assert all(v["position_error_m"]<2e-6 and v["orientation_error_rad"]<2e-5 for v in checked.values())
    result=json.loads((run/'result.json').read_text())
    assert result['status']=='completed' or (diagnostic_error_run and result['status']=='error')
    names=json.loads((run/'frame-names.json').read_text())
    meta=json.loads((run/'frame-geometry.json').read_text())
    all_links=(run/'robot-colliders.json').exists()
    grippers=json.loads((run/('robot-colliders.json' if all_links else 'gripper-colliders.json')).read_text())
    cooked_file=run/'robot-cooked-colliders.json'
    if all_links and cooked_file.exists():
        cooked=json.loads(cooked_file.read_text())
        proof=cooked.pop('_provenance')
        assert proof['source_run']==run.name and not proof['scene_modified']
        for name,expected in proof['source_sha256'].items():
            assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected
        for side in ('left','right'):
            assert cooked[side]['body_names']==grippers[side]['body_names']
            assert cooked[side]['indices']==grippers[side]['indices']
            assert [s['name'] for s in cooked[side]['colliders']]==[s['name'] for s in grippers[side]['colliders']]
        grippers=cooked
    scene=json.loads((run/'scene-audit.json').read_text())
    rows=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    chunks=[np.load(p) for p in sorted((run/'physics-chunks').glob('*.npz'))]
    bodies=np.concatenate([c['frame_body_poses'] for c in chunks])
    q=np.concatenate([c['frame_joint_positions'] for c in chunks])
    hz=request['physics_hz'];decimation=hz//20
    assert len(bodies)==len(q)==len(rows)*decimation
    assert len(rows)==result['control_steps'] and len(bodies)==result['physics_steps']
    assert np.isfinite(bodies).all() and np.isfinite(q).all(), 'Nonfinite native frame state'
    shapes=[]
    for shape in meta['colliders']:
        if shape['enabled']:
            prepared=prepare(shape,bodies[:,names['body_names'].index(shape['body'])])
            prepared['name']='Frame/'+shape['body']+'/'+shape['name']
            shapes.append(prepared)
    hands=[]
    for side,geometry in grippers.items():
        poses=np.concatenate([c[side+('_body_poses' if all_links else '_grasp_poses')] for c in chunks])
        assert len(poses)==len(bodies) and np.isfinite(poses).all(), 'Missing or nonfinite native robot state'
        for shape in geometry['colliders']:
            prepared=prepare(shape,poses[:,geometry['body_names'].index(shape['body'])])
            prepared['name']=side+shape['name']
            prepared['side']=side
            hands.append(prepared)
    adjacent={frozenset(p) for p in [('Bridge','Left'),('Bridge','Right'),('Bridge','Connector'),('Connector','Handle'),('Rocker','Clip')]}
    pairs=[]
    refinements=[]
    candidates=[]
    for a,b in itertools.combinations(shapes,2):
        pair=frozenset((a['body'],b['body']))
        if len(pair)>1 and pair not in adjacent:candidates.append((a,b))
    candidates.extend(itertools.product(shapes,hands))
    table_vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))*np.array(scene.get('table_size_m',station.TABLE_SIZE))/2
    table_pose=np.tile(np.r_[scene.get('table_center_m',station.TABLE_CENTER),[0.,0.,0.,1.]],(len(bodies),1))
    table=prepare(dict(name='Table',body='Table',points=table_vertices),table_pose)
    candidates.extend((shape,table) for shape in shapes)
    complete_robot_audit=all_links and request.get('robot_reopen',False)
    if complete_robot_audit:
        import xml.etree.ElementTree as ET
        from yam_arena import YAM_URDF
        robot_adjacent=set()
        for joint in ET.parse(YAM_URDF).getroot().findall('joint'):
            robot_adjacent.add(frozenset((joint.find('parent').get('link'),joint.find('child').get('link'))))
        candidates.extend((hand,table) for hand in hands if hand['body']!='base')
        for a,b in itertools.combinations(hands,2):
            same_arm=a['side']==b['side']
            linked=frozenset((a['body'],b['body']))
            if same_arm and (a['body']==b['body'] or linked in robot_adjacent):continue
            candidates.append((a,b))
    for i,(a,b) in enumerate(candidates):
        depth,steps,refinement=compare(a,b)
        if refinement is not None:refinements.append(refinement)
        if depth>1e-6:pairs.append(dict(first=a['name'],second=b['name'],max_depth_m=depth,violating_steps=steps))
        if i%200==0:
            progress=dict(pair=i,total=len(candidates),penetrating_pairs=sum(bool(p['violating_steps']) for p in pairs))
            (run/'frame-audit-heartbeat.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    joints=names['joint_names']
    wing=np.rad2deg(q[:,[joints.index('HingeLeft'),joints.index('HingeRight')]])
    contact_peak=max([p['max_depth_m'] for p in pairs],default=0.)
    gear_error=float(abs(wing.sum(1)).max())
    numerical_passed=contact_peak<=.001 and gear_error<=1.
    report = dict(
        scope='Every native rigid contact in a completed episode, independent of policy, phase labels and completion mode.',
        recording_status=result['status'], full_task_success=False,
        diagnostic_only=request.get('policy') == 'interface-check',
        complete_recorded_step_coverage=True, physics_steps=len(bodies), control_steps=len(rows),
        all_robot_links_against_frame_audited=all_links,
        robot_table_and_self_contacts_audited=complete_robot_audit,
        robot_geometry_source='native_cooked' if all_links and cooked_file.exists() else 'authored_convex_hull_outer_bound',
        max_convex_penetration_m=contact_peak, overlapping_pairs=pairs,
        capsule_projection_refinements=refinements, finite_native_rigid_state=True,
        max_gear_error_deg=gear_error, recorded_contact_checks_passed=numerical_passed,
        contact_checks_passed=numerical_passed, frame_handling_passed=False,
        assessor_source_sha256={name: digest(run / 'source' / name) for name in HELPERS},
        assessment_is_post_recording=True,
        limitations=[
            'No endpoint is inferred from controller phase names; the independent outcome scorer evaluates installation and retention.',
            'All collision candidates, capsule refinements, adjacent-link exclusions and 1 mm penetration limits are unchanged from the archived contact audit.',
            'The material-contact audit remains separate. No simulator state or launch source is modified.'])
    report['assessor_source_sha256'][assessor.name] = digest(assessor)
    report = helpers.write_report(run, report, all_links, cooked_file)
    for path in [assessor, assessor.with_name('task_paths.py'), *(run / 'source' / name for name in HELPERS)]:
        report['source_sha256'][str(path.relative_to(run))] = digest(path)
    assert digest(assessor) == digest(Path(__file__).resolve())
    (run / 'frame-robot-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    report = audit(args.run)
    print(json.dumps({key: report[key] for key in ('physics_steps', 'control_steps',
        'contact_checks_passed', 'max_convex_penetration_m', 'max_gear_error_deg',
        'robot_table_and_self_contacts_audited')}, indent=2))
