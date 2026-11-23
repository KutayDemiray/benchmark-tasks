"""Require completed physical and public-interface evidence before hosted evaluation."""

import base64
import hashlib
import io
import json

import numpy as np
from pathlib import Path

from physics_provenance import PHYSICAL_SOURCES, verify_physical_sources
from review_stability_round import review, verified_worker
from submission_preflight import interface_check, full_demonstration

HERE=Path(__file__).resolve().parent
SHARED_RUNTIME=('policy_worker.py','local_episode.py','tabletop_backend.py','policy_contract.py',
    'streaming_score.py','tabletop_scoring.py','tabletop_clearance.py','robot_scoring.py','fresh_workers.py')


def readiness(stability, validation, demo, local_check=None):
    """Recompute final evidence and require exactly the same native implementation."""
    gate=review(stability)
    assert gate['verification_pass'] and gate['gate_passed'], 'Completed stability gate required'
    proof=verify_physical_sources(HERE,stability/'task-bundle',json.loads((stability/'manifest.json').read_text()))
    interface=interface_check(validation)
    audit=verified_worker(demo)
    assert audit['verification_pass'] and len(audit['episodes'])==1
    setup=json.loads((demo/'episode-00/setup.json').read_text())
    result=json.loads((demo/'episode-00/result.json').read_text())
    full=full_demonstration(result,setup,gate['selected_configuration'])
    evaluation=json.loads((demo.parent/'evaluation.json').read_text())
    assert evaluation['mode']=='scripted' and evaluation['model_call'] is False
    assert evaluation['control_hz']==20 and evaluation['seconds']==300., 'Demo and model must share the 300-second action budget'
    sources=json.loads((demo/'manifest.json').read_text())['sources']
    for name in SHARED_RUNTIME:
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sources[name], ('Shared demo/model runtime changed',name)
    verify_physical_sources(demo/'task-bundle',stability/'task-bundle',json.loads((stability/'manifest.json').read_text()))
    checked=None
    if local_check is not None:
        assert json.loads((local_check/'exit.json').read_text())['returncode']==0
        checked=json.loads((local_check/'contract-checks.json').read_text())
        assert checked['passed'] is True and checked['hosted_calls']==0
        assert checked['executed_steps']==100 and checked['requests']==2
        assert checked['native_script_rejected'] is True and checked['native_step_during_inference']==0
        startup=checked['startup_camera_filter']
        assert startup['discarded_observations']==1 and startup['first_forwarded_time_s']==.05
        assert startup['forwarded_observation_calls']==99 and startup['additional_actions']==0
        local=verified_worker(local_check/'worker-00')
        assert local['verification_pass']
        assert not json.loads((local_check/'worker-00/episode-00/result.json').read_text())['invalid_reasons']
        local_setup=json.loads((local_check/'worker-00/episode-00/setup.json').read_text())
        assert local_setup['seed']==setup['seed'] and local_setup['controller']=='none' and local_setup['control_hz']==20
        for key in ('fabric','snap','simulation','tabletop','garment','completion'):
            assert local_setup['parameters'][key]==setup['parameters'][key], ('Local check configuration differs',key)
        local_sources=json.loads((local_check/'worker-00/manifest.json').read_text())['sources']
        for name in (*SHARED_RUNTIME,*PHYSICAL_SOURCES,'dream_execution.py','dream_readiness.py','dream_local_check.py'):
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==local_sources[name], ('Local interface code differs',name)
    return dict(ready=True,configuration=gate['selected_configuration'],stability=gate,source_proof=proof,
                interface=interface,demonstration=full,demonstration_path=str(demo),local_adapter_check=checked)



def audit_public_inputs(run, native_rows, steps, error):
    """Validate exact forwarded JPEG/state records after the one stale reset sample."""
    from PIL import Image
    from inspect_robots_dream import _station
    from policy_contract import CAMERAS, INSTRUCTION
    initial=json.loads((run/'initial-policy-state.json').read_text())
    assert initial['policy_context_forwarded'] is False
    first=json.loads((run/'first-forwarded-policy-state.json').read_text())
    assert first['step']==1 and first['time_s']==.05 and first['instruction']==INSTRUCTION
    assert np.array_equal(first['joint_pos'],native_rows[0]['joint_positions'])
    startup=json.loads((run/'startup-camera-filter.json').read_text())
    assert startup['discarded_observations']==1 and startup['discarded_time_s']==0.
    assert startup['first_forwarded_time_s']==.05 and startup['additional_actions']==0
    count=0
    with (run/'inference-observations.jsonl').open() as stream:
        for index,line in enumerate(stream,start=1):
            record=json.loads(line)
            assert set(record)=={'index','time_s','instruction','joint_pos','camera_jpeg_base64'}
            assert record['index']==index and abs(record['time_s']-index/20)<1e-8
            assert record['instruction']==initial['instruction']==INSTRUCTION
            assert index<=len(steps), 'More observations than executed actions plus one refused request'
            assert np.array_equal(record['joint_pos'],native_rows[index-1]['joint_positions']), 'Logged model input is not measured state'
            assert set(record['camera_jpeg_base64'])==set(CAMERAS)
            for name,encoded in record['camera_jpeg_base64'].items():
                raw=base64.b64decode(encoded,validate=True)
                with Image.open(io.BytesIO(raw)) as frame:
                    assert frame.format=='JPEG' and frame.mode=='RGB' and frame.size==(320,176)
                    frame.load()
                if index==1:
                    with Image.open(run/('first-forwarded-'+name+'.png')) as frame:
                        assert encoded==_station.jpeg(np.asarray(frame)), 'First forwarded camera identity/content differs'
            count+=1
    assert len(steps)-1<=count<=len(steps)-1+(1 if error is not None else 0)
    assert count==startup['forwarded_observation_calls']
    return dict(observations=count,cameras=list(CAMERAS),image_size_wh=[320,176],
        measured_state_verified=True,first_forwarded_camera_binding=True,public_input_whitelist_verified=True,
        startup_camera_filter=startup,jpeg_inputs=str(run/'inference-observations.jsonl'))


def audit_model_run(run, demo):
    """Independently verify a genuine recorded attempt without requiring model success."""
    assert json.loads((run/'exit.json').read_text())['returncode']==0
    evidence=json.loads((run/'model-evaluation.json').read_text())
    assert evidence['mode']=='dream_live' and evidence['model']=='dreamlabs/dream-1-beta'
    assert evidence['model_attempt'] is True and evidence['hosted_requests_started']>0
    assert evidence['episode_retries']==0 and evidence['model_fallback'] is False
    assert evidence['controller']=='none' and evidence['scripted_assistance'] is False
    assert evidence['control_hz']==20 and evidence['declared_action_budget']==6000
    worker=run/'worker-00';audit=verified_worker(worker)
    assert audit['verification_pass'] and len(audit['episodes'])==1
    result=json.loads((worker/'episode-00/result.json').read_text())
    assert result==evidence['result'] and result['task_score']==audit['episodes'][0]['task_score']
    actual=json.loads((worker/'episode-00/setup.json').read_text())
    reference=json.loads((demo/'episode-00/setup.json').read_text())
    assert actual['seed']==reference['seed'] and actual['control_hz']==reference['control_hz']==20
    assert actual['controller']=='none'
    for key in ('fabric','snap','simulation','tabletop','garment','completion'):
        assert actual['parameters'][key]==reference['parameters'][key], ('Demo/model configuration differs',key)
    source=json.loads((worker/'manifest.json').read_text())['sources']
    reference_source=json.loads((demo/'manifest.json').read_text())['sources']
    for name in (*SHARED_RUNTIME,*PHYSICAL_SOURCES):
        assert source[name]==reference_source[name], ('Demo/model implementation differs',name)
    events=[json.loads(line) for line in (run/'inference.jsonl').read_text().splitlines()]
    started=[e for e in events if e['event']=='started'];completed=[e for e in events if e['event']=='completed']
    assert len(started)==evidence['hosted_requests_started'] and len(completed)==evidence['hosted_requests_completed']
    assert all(event['chunk']['attempts']==1 for event in completed)
    assert [e['request'] for e in started]==list(range(len(started)))
    assert [e['request'] for e in completed]==list(range(len(completed)))
    assert len(completed)<=len(started)<=len(completed)+1
    steps=[json.loads(line) for line in (run/'policy-steps.jsonl').read_text().splitlines()]
    assert len(steps)==result['control_steps']<=6000
    assert all(row['native_physics_steps_during_inference']==0 for row in steps)
    transcript=json.loads((run/'policy-transcript.json').read_text())
    assert transcript['preroll_steps']==32 and transcript['rows_per_request']==48
    assert transcript['dry_run'] is False and transcript['model']==evidence['model']
    initial=json.loads((run/'initial-policy-state.json').read_text())
    targets=[initial['joint_pos']]*min(32,len(steps))
    for event in completed:
        targets.extend(event['chunk']['rows'][:48])
    assert len(targets)>=len(steps)
    native_rows=[json.loads(line) for line in (worker/'episode-00/trace.jsonl').read_text().splitlines()]
    assert len(native_rows)==len(steps)
    for index,(policy_row,native_row) in enumerate(zip(steps,native_rows,strict=True)):
        assert policy_row['step']==index and abs(policy_row['input_time_s']-index/20)<1e-8
        assert abs(policy_row['output_time_s']-(index+1)/20)<1e-8
        assert np.array_equal(policy_row['requested_action'],targets[index]), 'Action was not the model row or standard preroll'
        measured=initial['joint_pos'] if index==0 else native_rows[index-1]['joint_positions']
        assert np.array_equal(policy_row['input_joint_pos'],measured), 'Model state was not the measured native state'
        assert np.array_equal(native_row['requested_action'],policy_row['requested_action'])
    inputs=audit_public_inputs(run,native_rows,steps,evidence['error'])
    assert inputs['startup_camera_filter']==evidence['startup_camera_filter']
    return dict(verification_pass=True,model=evidence['model'],score=result['task_score'],public_inputs=inputs,
        termination_reason=result['termination_reason'],invalid_reasons=result['invalid_reasons'],
        inference_requests=len(started),completed_requests=len(completed),actions=len(steps),
        model_error=evidence['error'],source_bundle_sha256=evidence['source_bundle_sha256'],
        model_success_required=False,run=str(run),demonstration=str(demo))
