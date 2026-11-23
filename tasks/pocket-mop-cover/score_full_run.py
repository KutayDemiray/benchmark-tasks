"""Score a simulated robot-cover recording after its native validity audits.

Numerical strain/contact rejection is simulation-only; physical-trial judging
is specified separately in REAL_TRIAL_SCORING.md. The result API is unchanged.
"""

from task_paths import OUTPUT_ROOT

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from outcome_native import NativeOutcome


def score(run,workers=1,prefix=''):
    """Write reproducible native observations and a separate benchmark result."""
    request=json.loads((run/'request.json').read_text())
    assert request.get('robot_cover'), 'Laboratory and bare-frame runs cannot score'
    assert request.get('policy', 'scripted') != 'interface-check', 'Offline interface checks are not benchmark episodes'
    assert not request.get('release_probe_run'), 'Release probes cannot score as benchmark episodes'
    result=json.loads((run/'result.json').read_text())
    assert result['status']=='completed'
    from invalid_witness import strain_witness
    witness=strain_witness(run)
    if witness is not None:
        report=dict(score=0.,success=False,termination_reason='invalid',
            invalid_reasons=[witness['reason']],control_steps=result['control_steps'],
            native_counterexample=witness,
            assessment='A verified native threshold violation rejects the run; complete positive audits are unnecessary for this zero score.')
        (run/(prefix+'benchmark-result.json')).write_text(json.dumps(report,indent=2))
        return report
    cloth=json.loads((run/'full-cloth-audit.json').read_text())
    rigid=json.loads((run/'frame-robot-audit.json').read_text())
    assert cloth['complete'] and rigid['robot_table_and_self_contacts_audited']
    for report in (cloth,rigid):
        for name,expected in report['source_sha256'].items():
            assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected, f'Audit is stale: {name}'
    errors=[]
    if not cloth['numerical_checks_passed']:errors.append('cloth_numerical_audit')
    if not rigid['contact_checks_passed']:errors.append('rigid_contact_audit')
    proof={name:hashlib.sha256((run/name).read_bytes()).hexdigest()
        for name in ('full-cloth-audit.json','frame-robot-audit.json')}
    if errors:
        report=dict(score=0.,success=False,termination_reason='invalid',invalid_reasons=errors,
            control_steps=result['control_steps'],audit_sha256=proof)
        (run/(prefix+'benchmark-result.json')).write_text(json.dumps(report,indent=2))
        return report
    geometry=np.load(run/'geometry.npz')
    scene=json.loads((run/'scene-audit.json').read_text())
    assert not scene['laboratory_fixture'] and not scene['cloth_attachments']
    assert not scene.get('diagnostic_initialization'), 'Diagnostic initializations cannot score'
    robots=json.loads((run/'robot-cooked-colliders.json').read_text())
    numerical=dict(finite=cloth['finite'],max_edge_strain=cloth['max_abs_edge_strain'],
        max_penetration_m=max(.001,rigid['max_convex_penetration_m']),max_seam_gap_m=cloth['max_seam_gap_m'],
        unexpected_constraint=False,diagnostic_fixture=False)
    table_center=np.array(scene['table_center_m'][:2]);table_half=np.array(scene['table_size_m'][:2])/2
    observer=NativeOutcome(geometry['initial'],geometry['faces'],json.loads((run/'cover-mesh.json').read_text()),
        json.loads((run/'frame-geometry.json').read_text()),json.loads((run/'frame-names.json').read_text()),
        robots,float(scene['table_center_m'][2]+scene['table_size_m'][2]/2),request['physics_hz'],numerical,
        (table_center-table_half,table_center+table_half))
    rows=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    decimation=request['physics_hz']//20
    count=0;maximum=0.;first_success=None;final=None
    with (run/(prefix+'outcomes.jsonl')).open('w') as output:
        if workers>1:
            from parallel_outcomes import observations
            from score_logic import OutcomeScorer
            reducer=OutcomeScorer()
            for observation,measurements in observations(run,numerical,workers,prefix):
                final={**reducer.update(observation),**measurements}
                maximum=max(maximum,final['score'])
                if final['success'] and first_success is None:first_success=count
                output.write(json.dumps({**rows[count],**final,'full_task_success':final['success'],
                    'score_status':'Assessed from complete independent native audits',
                    'observation':asdict(observation)})+'\n')
                count+=1
        else:
            count,maximum,first_success,final=sequential_outcomes(run,observer,rows,decimation,output)
    assert count==result['control_steps'] and final is not None
    report=dict(score=final['score'],maximum_score=maximum,success=final['success'],
        termination_reason='success' if final['success'] else 'incomplete',invalid_reasons=final['invalid_reasons'],
        control_steps=count,first_success_step=first_success,audit_sha256=proof,
        observations_sha256=hashlib.sha256((run/(prefix+'outcomes.jsonl')).read_bytes()).hexdigest(),
        assessment='Pocket enclosure sampled at 20 Hz; numerical limits and support/clearance bounds cover every native substep.',
        clearance_method='Swept actual convex geometry with conservative separating projections; swept triangle boxes for cloth. Every recorded native substep is included; unproved separation remains possible support.',
        scorer_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('score_full_run.py','outcome_native.py','score_logic.py','measurements.py','audit_mechanism.py')})
    (run/(prefix+'benchmark-result.json')).write_text(json.dumps(report,indent=2))
    return report


def sequential_outcomes(run,observer,rows,decimation,output):
    """Retain the original sequential computation as the reference backend."""
    count=0;maximum=0.;first_success=None;final=None
    for path in sorted((run/'physics-chunks').glob('*.npz')):
        with np.load(path) as chunk:
            length=len(chunk['positions'])
            assert length%decimation==0
            fields=('positions','velocities','frame_body_poses','frame_joint_positions','left_body_poses','right_body_poses')
            recorded={name:chunk[name] for name in fields}
            for start in range(0,length,decimation):
                state={name:recorded[name][start:start+decimation] for name in fields}
                observation,final=observer.update(count,state)
                maximum=max(maximum,final['score'])
                if final['success'] and first_success is None:first_success=count
                output.write(json.dumps({**rows[count],**final,'full_task_success':final['success'],
                    'score_status':'Assessed from complete independent native audits',
                    'observation':asdict(observation)})+'\n')
                count+=1
    return count,maximum,first_success,final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--workers',type=int,choices=range(1,9),default=1)
    parser.add_argument('--output-prefix',choices=('', 'parallel-check-', 'refined-clearance-'),default='')
    args=parser.parse_args()
    assert args.run.resolve().parent==OUTPUT_ROOT
    print(json.dumps(score(args.run,args.workers,args.output_prefix),indent=2))
