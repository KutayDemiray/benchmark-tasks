"""Independent contact and retention measurements for a stock-robot frame test."""

from task_paths import OUTPUT_ROOT
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_mechanism import features,penetration,capsule_proxy


def prepare(shape,poses):
    vertices=capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
    normals,edges=features(vertices)
    rotations=Rotation.from_quat(poses[:,3:7]).as_matrix()
    low,high=vertices.min(0),vertices.max(0)
    corners=np.array(list(itertools.product(*zip(low,high))))
    world=np.einsum('tij,pj->tpi',rotations,corners)+poses[:,None,:3]
    return dict(name=shape.get('name',''),body=shape['body'],capsule=shape.get('capsule'),
        data=(poses[:,:3],rotations,vertices,normals,edges),low=world.min(1),high=world.max(1),
        global_low=world.min((0,1)),global_high=world.max((0,1)))


def capsule_projection_bound(capsule_shape,polyhedron,indices):
    """Bound penetration using exact capsule projections on polyhedron normals.

    Separation along any tested axis certifies separation of the real shapes.
    Using only these axes may overestimate depth, but cannot underestimate it.
    This resolves overly conservative inflated-polyhedron capsule failures.
    """
    capsule=capsule_shape['capsule']
    cpos,crot,_,_,_=capsule_shape['data']
    ppos,prot,vertices,normals,_=polyhedron['data']
    axes=np.einsum('tij,pj->tpi',prot[indices],normals)
    world=np.einsum('tij,pj->tpi',prot[indices],vertices)+ppos[indices,None]
    projection=np.einsum('tvi,tai->tav',world,axes)
    center=cpos[indices]+np.einsum('tij,j->ti',crot[indices],capsule['center'])
    middle=np.einsum('tai,ti->ta',axes,center)
    extent=capsule['height']/2*abs(np.einsum('tai,ti->ta',axes,crot[indices,:,2]))+capsule['radius']
    depth=np.minimum(middle+extent-projection.min(2),projection.max(2)-(middle-extent))
    return np.maximum(depth.min(1),0.)


def compare(a,b):
    if not (np.all(a['global_high']>=b['global_low']) and np.all(b['global_high']>=a['global_low'])):
        return 0.,[],None
    broad=np.all(a['high']>=b['low'],axis=1)&np.all(b['high']>=a['low'],axis=1)
    indices=np.flatnonzero(broad)
    if not len(indices):return 0.,[],None
    aa=a['data'];bb=b['data']
    depths=penetration((aa[0][indices],aa[1][indices],*aa[2:]),(bb[0][indices],bb[1][indices],*bb[2:]),upper_bound_tolerance=.001)
    refinement=None
    failed=depths>.001
    if np.any(failed) and bool(a.get('capsule'))!=bool(b.get('capsule')):
        capsule,polyhedron=(a,b) if a.get('capsule') else (b,a)
        exact_bound=capsule_projection_bound(capsule,polyhedron,indices[failed])
        refinement=dict(first=a['name'],second=b['name'],physics_steps=indices[failed].tolist(),
            inflated_proxy_peak_m=float(depths[failed].max()),
            analytic_projection_peak_bound_m=float(exact_bound.max()))
        depths[failed]=np.minimum(depths[failed],exact_bound)
    return float(depths.max()),indices[depths>.001].tolist(),refinement


def write_report(run,report,all_links,cooked_file):
    """Bind the rigid measurements to their original recording and outcome status."""
    report['source_sha256']={}
    for path in [*sorted((run/'physics-chunks').glob('[0-9][0-9][0-9][0-9][0-9].npz')),
                 *[run/name for name in ('request.json','manifest.json','result.json','frame-geometry.json','frame-names.json',
                     'scene-audit.json','steps.jsonl','robot-colliders.json' if all_links else 'gripper-colliders.json')]]:
        report['source_sha256'][str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if cooked_file.exists():report['source_sha256'][cooked_file.name]=hashlib.sha256(cooked_file.read_bytes()).hexdigest()
    (run/'frame-robot-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def audit(run,diagnostic_error_run=False):
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
    if request.get('release_probe_run') and result['status']=='completed':
        report=dict(scope='Every recorded rigid contact in a cold-start release diagnostic; no full-task retention claim.',
            recording_status=result['status'],full_task_success=False,diagnostic_only=True,
            benchmark_eligible=False,complete_recorded_step_coverage=True,
            physics_steps=len(bodies),control_steps=len(rows),
            all_robot_links_against_frame_audited=all_links,
            robot_table_and_self_contacts_audited=complete_robot_audit,
            robot_geometry_source='native_cooked' if all_links and cooked_file.exists() else 'authored_convex_hull_outer_bound',
            max_convex_penetration_m=contact_peak,overlapping_pairs=pairs,
            capsule_projection_refinements=refinements,finite_native_rigid_state=True,
            max_gear_error_deg=gear_error,recorded_contact_checks_passed=numerical_passed,
            contact_checks_passed=numerical_passed,frame_handling_passed=False,
            assessor_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('audit_frame_robot.py','audit_mechanism.py','sat_accelerated.py')},
            limitations=['Initialization is a recorded held pose with a cold solver and zero velocities.',
                'This recording omits installation and functional retention; benchmark scoring is forbidden.',
                'All rigid contact checks are unchanged. Material contact requires a separate audit.'])
        return write_report(run,report,all_links,cooked_file)
    if result['status']!='completed':
        report=dict(scope='Diagnostic rigid geometry of every recorded step in a stopped controller; material audit remains separate.',
            recording_status=result['status'],full_task_success=False,diagnostic_only=True,
            complete_recorded_step_coverage=True,physics_steps=len(bodies),control_steps=len(rows),
            all_robot_links_against_frame_audited=all_links,
            robot_table_and_self_contacts_audited=complete_robot_audit,
            robot_geometry_source='native_cooked' if all_links and cooked_file.exists() else 'authored_convex_hull_outer_bound',
            max_convex_penetration_m=contact_peak,overlapping_pairs=pairs,
            capsule_projection_refinements=refinements,
            finite_native_rigid_state=True,max_gear_error_deg=gear_error,
            recorded_contact_checks_passed=numerical_passed,contact_checks_passed=False,
            frame_handling_passed=False,
            limitations=['Stopped-controller diagnostics cannot certify a benchmark score.',
                'Conservative SAT bounds; failed capsule proxies refined with exact capsule projections.',
                'Fixed mounting bases and adjacent articulated links are excluded.'])
        return write_report(run,report,all_links,cooked_file)
    unlock_only=request.get('robot_unlock',False) and not request.get('robot_reopen',False)
    hold_phases=('unlocked_hold',) if unlock_only else ('hold_flat_frame','translate_frame','return_frame','retention')
    hold_controls=[i for i,row in enumerate(rows) if row['phase'] in hold_phases]
    hold=np.concatenate([np.arange(i*decimation,(i+1)*decimation) for i in hold_controls])
    bridge=bodies[:,names['body_names'].index('Bridge'),:3]
    initial=bridge[8*hz:9*hz].mean(0)
    lifted=bridge[hold,2]-initial[2]
    table_top=table_pose[0,2]+float(np.asarray(scene.get('table_size_m',station.TABLE_SIZE))[2])/2
    head_clearance=np.full(len(hold),np.inf)
    for shape in shapes:
        if shape['body'] in ('Left','Right'):
            position,rotation,vertices,_,_=shape['data']
            world=np.einsum('tij,pj->tpi',rotation[hold],vertices)+position[hold,None]
            head_clearance=np.minimum(head_clearance,world[:,:,2].min(1)-table_top)
    assert np.isfinite(head_clearance).all()
    max_angle=float(abs(wing[hold]).max())
    released_controls=[i for i,row in enumerate(rows) if row['phase']=='released']
    assert len(released_controls)>=40, 'Need two recorded seconds after release'
    released=np.concatenate([np.arange(i*decimation,(i+1)*decimation) for i in released_controls[-40:]])
    released_angle=float(abs(wing[released]).max())
    released_head_height=float((bridge[released,2]-table_top).max())
    hand_clearance=float('inf')
    for head in shapes:
        for hand in hands:
            gap=np.maximum(np.maximum(head['low'][released]-hand['high'][released],
                                      hand['low'][released]-head['high'][released]),0.)
            hand_clearance=min(hand_clearance,float(np.linalg.norm(gap,axis=1).min()))
    retained=float(head_clearance.min())>.10 and max_angle<=3.
    released_flat=released_angle<=3. and released_head_height<=.01
    holds=[i for i,row in enumerate(rows) if row['phase']==('unlocked_hold' if unlock_only else 'hold_flat_frame')]
    anchor=bridge[(holds[-1]+1)*decimation-1,:2]
    displacement=np.linalg.norm(bridge[hold,:2]-anchor,axis=1)
    report=dict(scope='Free rigid frame handled by stock robot; no textile or benchmark score.',
        full_task_success=False,diagnostic_only=True,physics_steps=len(bodies),control_steps=len(rows),
        all_robot_links_against_frame_audited=all_links,
        robot_table_and_self_contacts_audited=complete_robot_audit,
        robot_geometry_source='native_cooked' if all_links and cooked_file.exists() else 'authored_convex_hull_outer_bound',
        max_convex_penetration_m=contact_peak,overlapping_pairs=pairs,
        capsule_projection_refinements=refinements,
        held_wing_angle_deg=max_angle,min_held_bridge_lift_m=float(lifted.min()),
        min_held_head_clearance_m=float(head_clearance.min()),
        held_head_xy_excursion_m=float(displacement.max()),
        returned_head_xy_error_m=float(np.linalg.norm(bridge[hold[-1],:2]-anchor)),
        final_two_seconds_wing_angle_deg=released_angle,
        final_two_seconds_head_height_m=released_head_height,
        final_two_seconds_gripper_clearance_lower_bound_m=hand_clearance,
        finite_native_rigid_state=True,max_gear_error_deg=gear_error,
        contact_checks_passed=numerical_passed,
        handle_lift_and_retention_observed=retained,
        released_flat_and_clear=released_flat and hand_clearance>=.02,
        frame_handling_passed=numerical_passed and retained and released_flat and hand_clearance>=.02,
        limitations=['Capsule uses a conservative outer proxy; failed pairs are refined with exact capsule projections, still conservative.',
            'SAT stops once a separating-axis depth upper bound is below 1 mm; reported depths are conservative bounds.',
            'Robot/table and nonadjacent robot/self contacts included; fixed mounting bases and adjacent URDF links excluded.' if complete_robot_audit else
            'All robot link/frame contacts included; robot/table and robot/self contacts are not audited.' if all_links else 'Only authored gripper/tip and frame colliders included; robot arm link collisions are not fully audited.',
            'Starts freely folded; robot handle press tested, no pocket installation.' if request.get('frame_press',False) else 'Starts flat and contact-locked. Robot pedal unlocking and pocket installation are not tested.'])
    if unlock_only:
        minimum_fold=float(abs(wing[hold]).min())
        report.update(scope='Stock robot physical pedal unlock and handle lift; no textile or benchmark score.',
            minimum_unlocked_hold_fold_deg=minimum_fold,
            robot_unlock_passed=numerical_passed and minimum_fold>=30.,frame_handling_passed=False)
        report['limitations'][-1]='Physical pedal operation tested; pocket installation remains untested.'
    if request.get('robot_reopen',False):
        folded_controls=[i for i,row in enumerate(rows) if row['phase']=='settle_folded_frame']
        folded=np.concatenate([np.arange(i*decimation,(i+1)*decimation) for i in folded_controls])
        minimum_fold=float(abs(wing[folded]).min())
        report.update(scope='Stock robot pedal unlock, inner-rail reopening, passive locking and retention; no textile.',
            minimum_initial_fold_deg=minimum_fold,
            physical_reopening_passed=report['frame_handling_passed'] and minimum_fold>=30.)
        report['limitations'][-1]='Physical unlocking and reopening tested; pocket installation remains untested.'
    if request.get('robot_cover',False):
        report['scope']='All rigid bodies during two-pocket robot integration; material contact audit remains separate.'
        report['limitations'][-1]='Rigid head handling alone cannot establish cover installation or a benchmark score.'
    return write_report(run,report,all_links,cooked_file)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--diagnostic-error-run',action='store_true',help='Inspect recorded contacts without retention or score certification')
    args=p.parse_args();assert args.run.resolve().parent==OUTPUT_ROOT
    r=audit(args.run,args.diagnostic_error_run)
    r['overlapping_pairs']=[{**p,'violating_steps':len(p['violating_steps'])} for p in r['overlapping_pairs']]
    print(json.dumps(r,indent=2))
