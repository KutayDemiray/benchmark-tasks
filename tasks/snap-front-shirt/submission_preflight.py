"""Audit local submission evidence without launching a model or publishing anything."""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from bench import write
from physics_provenance import verify_physical_sources
from task_paths import OUTPUT_ROOT

HERE = Path(__file__).resolve().parent
CAMERAS = ('top_cam', 'left_cam', 'right_cam')
PUBLIC_FIELDS = ('title', 'category', 'instruction', 'setup', 'scoring',
                 'run_limit', 'fidelity', 'scores')
CATEGORIES = ('Impact on jobs', 'Impact on AI timelines', 'Safety and alignment',
              'Societal relatability', 'Other')


class PendingEvidence(Exception):
    """An expected completed artifact has not been produced yet."""


def read(path):
    if not path.is_file():
        raise PendingEvidence(str(path))
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def full_demonstration(result, setup, configuration):
    """Require a valid full three-snap outcome in the selected physical profile."""
    assert result['stage'] == 'tabletop_task'
    assert result['full_task_success'] is True and result['task_score'] == 1.
    assert result['retained_pairs'] == [0, 1, 2] and not result['invalid_reasons']
    assert result['original_hanger_task_success'] is False
    assert setup['case'] == 'fasten_three' and setup['action_dimension'] == 14
    cfg = setup['parameters']
    assert setup['control_hz'] == cfg['completion']['control_hz'] == 20
    assert cfg['completion']['run_limit_seconds'] == 300
    assert 0 < result['control_steps'] <= 6000
    assert result['physics_steps_observed'] == result['control_steps'] * 48 + 1
    assert cfg['tabletop']['required_snap_indices'] == [0, 1, 2]
    assert cfg['fabric']['solver_iterations'] == configuration['iterations']
    assert cfg['garment']['body_mesh_pitch_m'] == configuration['pitch']
    assert cfg['fabric']['contact_offset_m'] == .001
    assert cfg['simulation']['physics_hz'] == 960
    assert setup['recorded_cameras'] is True
    return dict(score=1.,controls=result['control_steps'],seconds=result['control_steps']/20,
                seed=setup['seed'],configuration=configuration)


def media_binding(manifest, worker, native_manifest, result, setup):
    """An old success cannot substitute for footage of the selected demonstration."""
    assert Path(manifest['source_run']).resolve() == worker.resolve(), 'Different source run'
    assert manifest['source_episode'] == 'episode-00'
    assert manifest['source_bundle_sha256'] == native_manifest['bundle_sha256']
    assert manifest['source_seed'] == setup['seed']
    assert manifest['source_score'] == result['task_score'] == 1.
    assert manifest['source_controls'] == result['control_steps']
    assert manifest['model_attempt'] is False, 'A scripted demo is not a model attempt'
    assert manifest['temporal_edits'] is False and manifest['playback_speed'] == 1.
    assert manifest['images_synthesized'] is False
    assert set(manifest['source_cameras']) == set(CAMERAS)
    for name in CAMERAS:
        source = manifest['source_cameras'][name]
        assert source['decoded_frames'] == result['control_steps']
        assert source['fps'] == setup['control_hz'] and source['decode_pass'] is True
    demo = manifest['demo']
    assert demo['frames'] == result['control_steps'] and demo['fps'] == setup['control_hz']
    assert demo['decoded'] is True and 'h264' in demo['codec']
    assert 0 < demo['bytes'] <= 50_000_000



def replay_action_match(source, replay):
    """Check recorded native actions without extending or repairing an exhausted replay."""
    assert source.ndim == replay.ndim == 2 and source.shape[1] == replay.shape[1] == 14
    assert np.isfinite(source).all() and np.isfinite(replay).all()
    assert 0 < len(replay) <= len(source), 'Replay is empty or has appended actions'
    assert np.array_equal(replay,source[:len(replay)]), 'Replay actions differ from the source prefix'
    return dict(recorded_actions=len(source),replayed_actions=len(replay),
                exact_executed_prefix=True,extra_actions=0)


def public_metadata(task):
    """Check public field structure and obvious identifying path leaks."""
    for name in PUBLIC_FIELDS:
        assert isinstance(task.get(name), str) and task[name].strip(), name
        for forbidden in ('/home/', '/Users/', 'kutay', 'ROBOCURVE_TOKEN'):
            assert forbidden.lower() not in task[name].lower(), ('Identifying text', name)
    assert task['category'] in CATEGORIES
    return dict(text_fields_present=True,obvious_identifiers_absent=True,
                media_present=isinstance(task.get('media'),str) and bool(task['media'].strip()),
                scope='Structure and obvious text identifiers only; human review remains required')


class Review:
    """Collect passed, failed and pending checks without promoting missing evidence."""

    def __init__(self):
        self.checks = []

    def check(self, name, function):
        try:
            evidence = function()
            self.checks.append(dict(name=name,status='passed',evidence=evidence))
            return evidence
        except PendingEvidence as exc:
            self.checks.append(dict(name=name,status='pending',detail=str(exc)))
        except (AssertionError, KeyError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.checks.append(dict(name=name,status='failed',detail=repr(exc)))
        return None


def native_stage(queue, name, mode):
    """Independently verify completed native workers; never consume a live trace."""
    from review_stability_round import verified_worker
    stage = queue/name
    assert not (stage/'failure.json').exists(), 'Native stage recorded a failure'
    assert read(stage/'exit.json')['returncode'] == 0, 'Native stage exited unsuccessfully'
    evaluation = read(stage/'evaluation.json')
    assert evaluation['mode'] == mode and evaluation['model_call'] is False
    assert evaluation['control_hz'] == 20
    read(stage/'results.json')
    workers = sorted(stage.glob('worker-*'))
    assert len(workers) == evaluation['episodes'] and workers
    records = []
    for worker in workers:
        assert read(worker/'exit.json')['returncode'] == 0
        assert not (worker/'failure.json').exists()
        audit = verified_worker(worker)
        assert audit['verification_pass'] is True and len(audit['episodes']) == 1
        result = read(worker/'episode-00/result.json')
        records.append(dict(worker=str(worker),result=result,
                            source_bundle_sha256=audit['source_bundle_sha256']))
    return dict(stage=str(stage),mode=mode,workers=records)


def interface_check(queue):
    stage = native_stage(queue,'interface','hold')
    assert len(stage['workers']) == 2
    pids = []
    for item in stage['workers']:
        worker = Path(item['worker'])
        result = item['result']
        assert not result['invalid_reasons'] and result['control_steps'] == 20
        checks = read(worker/'interface-checks.json')
        assert checks['invalid_action_rejected'] is True and checks['joints_unchanged'] is True
        assert checks['observation_time_advanced'] == checks['native_physics_steps_advanced'] == 0
        pids.append(read(worker/'exit.json')['worker_pid'])
    assert len(set(pids)) == 2, 'Reset reused one native process'
    return dict(stage=str(queue/'interface'),worker_pids=pids,control_hz=20,passed=True)


def baseline_check(queue, stability, configuration):
    from compare_tabletop_runs import compare
    stage = native_stage(queue,'baseline-reproduction','scripted')
    assert len(stage['workers']) == 2
    manifest = read(stability/'manifest.json')
    for item in stage['workers']:
        worker = Path(item['worker'])
        setup = read(worker/'episode-00/setup.json')
        full_demonstration(item['result'],setup,configuration)
        verify_physical_sources(worker/'task-bundle',stability/'task-bundle',manifest)
    comparison = compare(*(Path(item['worker']) for item in stage['workers']))
    assert comparison['passed'] is True
    return comparison


def media_check(media, worker):
    import imageio.v2 as imageio
    from PIL import Image
    from prepare_review_media import inspect_video
    manifest = read(media/'media-manifest.json')
    result = read(worker/'episode-00/result.json')
    setup = read(worker/'episode-00/setup.json')
    media_binding(manifest,worker,read(worker/'manifest.json'),result,setup)
    for name in CAMERAS:
        assert digest(worker/'episode-00'/(name+'.mp4')) == manifest['source_cameras'][name]['sha256']
    actual = inspect_video(media/'demo.mp4')
    for key in ('sha256','frames','fps','bytes'):
        assert actual[key] == manifest['demo'][key], ('Media differs',key)
    reader = imageio.get_reader(media/'demo.mp4')
    info = reader.get_meta_data()
    reader.close()
    assert 'yuv420p' in info['pix_fmt'], 'Review video pixel format differs'
    assert digest(media/'setup.png') == manifest['setup_image']['sha256']
    with Image.open(media/'setup.png') as image:
        image.verify()
    if manifest.get('visual_review_pending') is not False or not manifest.get('visual_review'):
        raise PendingEvidence('Inspect the overhead and both wrist views at start, middle and end')
    return dict(media=str(media),source_worker=str(worker),demo=actual,
                setup_time_s=manifest['setup_image']['time_s'],visual_review=manifest['visual_review'])


def audit(queue, stability, media=None, task_dir=HERE, model_run=None, interface_queue=None):
    """Review local evidence and explicitly leave external submission obligations open."""
    from review_stability_round import review as stability_review
    review = Review()
    task = read(task_dir/'task.json')
    review.check('public_metadata',lambda: public_metadata(task))
    progress = read(queue/'progress.json')
    def gate_check():
        read(stability/'exit.json')
        gate = stability_review(stability)
        assert gate['verification_pass'] is True and gate['gate_passed'] is True
        verify_physical_sources(task_dir,stability/'task-bundle',read(stability/'manifest.json'))
        return gate
    gate = review.check('completed_stability_gate',gate_check)
    review.check('native_20hz_interface_and_fresh_resets',lambda: interface_check(interface_queue or queue))
    def baseline():
        if gate is None:
            raise PendingEvidence('Selected complete stability evidence is required')
        return baseline_check(queue,stability,gate['selected_configuration'])
    reproduced = review.check('full_three_snap_demonstration_and_reproduction',baseline)
    worker = queue/'baseline-reproduction/worker-00'
    def shortcuts():
        if reproduced is None:
            raise PendingEvidence('Audited full-score baseline is required')
        from check_tabletop_shortcuts import check
        with contextlib.redirect_stdout(io.StringIO()):
            checked = check(worker)
        assert checked['passed'] is True and len(checked['cases']) == 13
        return checked
    review.check('scorer_shortcuts_on_selected_record',shortcuts)
    def replay():
        if reproduced is None:
            raise PendingEvidence('Audited source demonstration is required')
        evidence = native_stage(queue,'action-replay','action_replay')
        evaluation = read(queue/'action-replay/evaluation.json')
        assert Path(evaluation['replay_episode']).resolve() == (worker/'episode-00').resolve()
        assert len(evidence['workers']) == 1
        outcome = evidence['workers'][0]['result']
        replay_worker = Path(evidence['workers'][0]['worker'])
        assert evidence['workers'][0]['source_bundle_sha256'] == read(worker/'manifest.json')['bundle_sha256']
        source_setup = read(worker/'episode-00/setup.json')
        replay_setup = read(replay_worker/'episode-00/setup.json')
        assert source_setup['seed'] == replay_setup['seed']
        assert source_setup['control_hz'] == replay_setup['control_hz']
        for key in ('fabric','snap','simulation','tabletop','garment','completion'):
            assert source_setup['parameters'][key] == replay_setup['parameters'][key], ('Replay configuration differs',key)
        with np.load(worker/'episode-00/control-steps.npz') as source_data, \
             np.load(replay_worker/'episode-00/control-steps.npz') as replay_data:
            evidence['action_comparison'] = replay_action_match(source_data['actions'],replay_data['actions'])
        assert not outcome['invalid_reasons'], 'Native action replay produced invalid physics'
        evidence['exact_playback_success_required'] = False
        return evidence
    review.check('independent_native_action_replay',replay)
    def media_review():
        if reproduced is None:
            raise PendingEvidence('Media must bind to the selected audited demonstration')
        return media_check(media or queue/'validated-review-media',worker)
    review.check('selected_demonstration_media',media_review)
    def metadata_profile():
        if reproduced is None:
            raise PendingEvidence('Update the published profile after native 20 Hz validation')
        declared = task['validation_basis']['active_completion']
        assert declared['control_hz'] == 20 and declared['run_limit_seconds'] == 300
        assert declared['action_dimension'] == 14
        assert '20 Hz' in task['run_limit'] and '6,000' in task['run_limit'].replace('6000','6,000')
        return dict(control_hz=20,action_budget=6000,run_limit_seconds=300)
    review.check('published_control_profile',metadata_profile)
    model_audit=None
    if model_run is not None:
        from dream_readiness import audit_model_run
        model_audit=review.check('genuine_model_attempt',lambda: audit_model_run(model_run,worker))
    pending_external = [
        dict(item='Actual model attempt',status='verified' if model_audit is not None else 'not_audited_by_this_offline_tool',
             reason='A scripted or fake-session success cannot stand in for the required model attempt'),
        dict(item='Board novelty and ownership review',status='not_audited',
             reason='Public keyword screening does not verify the signed-in board'),
        dict(item='Uploaded matching media ID',status='not_authorized_or_audited',
             reason='No upload is performed or authenticated by this offline tool')]
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        queue=str(queue),queue_phase=progress['phase'],stability_run=str(stability),
        checks=review.checks,local_validation_ready=all(c['status']=='passed' for c in review.checks),
        submission_ready=False,external_requirements=pending_external,
        bimanual_status=dict(repeatable_one_snap_revision=progress.get('repeatable_one_snap_bimanual_revision'),
            full_task_success=progress.get('bimanual_full_task_success',False),
            required_by_reviewed_public_criteria=False),
        statistics=dict(required_success_rate=None,scope='No 5/5 or 9/10 quota imposed'),
        model_calls=False,uploads=False,
        note='Model audit checks recorded live-session/action provenance, not server-side attestation. Board eligibility and uploads remain unaudited. '
             'No local pass is a claim of RoboCurve acceptance.',
        auditor_sha256=digest(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--media',type=Path)
    parser.add_argument('--model-run',type=Path)
    parser.add_argument('--interface-queue',type=Path,help='Reuse a completed independently audited interface stage')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    paths = [args.queue,args.stability_run,args.out]
    if args.model_run is not None:
        paths.append(args.model_run)
    if args.interface_queue is not None:
        paths.append(args.interface_queue)
    if args.media is not None:
        paths.append(args.media)
    assert all(path.resolve().is_relative_to(OUTPUT_ROOT) for path in paths)
    assert not args.out.exists(), 'Preserve earlier preflight records; use a fresh output filename'
    result = audit(args.queue.resolve(),args.stability_run.resolve(),
                   None if args.media is None else args.media.resolve(),
                   model_run=None if args.model_run is None else args.model_run.resolve(),
                   interface_queue=None if args.interface_queue is None else args.interface_queue.resolve())
    write(args.out,result)
    print(json.dumps(dict(report=str(args.out),local_validation_ready=result['local_validation_ready'],
                         submission_ready=False,checks=[dict(name=c['name'],status=c['status']) for c in result['checks']])))


if __name__ == '__main__':
    main()
