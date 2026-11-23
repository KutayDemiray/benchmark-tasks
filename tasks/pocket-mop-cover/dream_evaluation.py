"""One hosted Dream episode, or an explicitly offline interface diagnostic."""

from task_paths import OUTPUT_ROOT

import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evaluation_contract import (ACTION_BUDGET, CAMERAS, CONTROL_HZ, HEIGHT, WIDTH,
    configuration_identity, embodiment_info, preserved_runtime_paths, process_action,
    public_observation, task_specification, vector, assessment_paths, validate_task_revision)

OUT = OUTPUT_ROOT
MODEL = 'dreamlabs/dream-1-beta'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def credential_status():
    """Read only configuration; never authenticate, print or store credential values."""
    from dream import credentials
    value = credentials.load()
    return dict(credential_file=str(value.path), credential_file_exists=value.path.is_file(),
                api_key_present=bool(value.api_key), network_calls=0)


def require_hosted_evidence(request, source):
    """Reject hosted calls until a matched full demonstration and replay pass."""
    revision = validate_task_revision(request, source)
    _, result_file, observations_file = assessment_paths(request)
    assert request['seconds'] == 120 and request['robot_cover']
    expected = configuration_identity(request)
    verified = {}
    baseline = None
    runtime = {str(path): sha256(path) for path in preserved_runtime_paths()}
    for key in ('demo_run', 'replay_evidence', 'interface_evidence'):
        name = request.get(key)
        assert name and Path(name).name == name and name not in ('.', '..'), 'Missing ' + key
        run = OUT / name
        req = json.loads((run / 'request.json').read_text())
        validate_task_revision(req, run / 'source')
        assert configuration_identity(req) == expected, 'Configuration differs: ' + key
        assert task_specification(run / 'source') == task_specification(source), 'Task instruction/specification changed'
        assert json.loads((run / 'preserved-sources.json').read_text()) == runtime, 'Installed runtime changed'
        matched = {name: json.loads((run / name).read_text()) for name in
            ('versions.json', 'scene-audit.json', 'effective-parameters.json', 'frame-geometry.json', 'frame-dynamics.json')}
        with np.load(run / 'geometry.npz') as archive:
            geometry = {name: archive[name].copy() for name in archive.files}
        with np.load(run / 'reset-initial-state.npz') as archive:
            reset = {name: archive[name].copy() for name in archive.files}
        if baseline is None:
            baseline = matched, geometry, reset
        else:
            assert matched == baseline[0], 'Native scene, parameters or versions differ'
            for current, original in ((geometry, baseline[1]), (reset, baseline[2])):
                assert set(current) == set(original)
                assert all(np.allclose(current[name], original[name], atol=2e-6, rtol=0.) for name in original), 'Initial state/geometry differs'
        result = json.loads((run / 'result.json').read_text())
        assert result['status'] == 'completed'
        assert json.loads((run / 'process.json').read_text())['returncode'] == 0
        assert json.loads((run / 'video-verification.json').read_text())['passed']
        assert json.loads((run / 'gpu-verification.json').read_text())['verified']
        manifest = json.loads((run / 'manifest.json').read_text())['files']
        # Controller-only changes do not authorize a changed simulator, reset,
        # public interface, numerical model, scorer or recording implementation.
        protected = ('task_paths.py', 'frame_robot.py', 'robot_session.py', 'full_cover.py', 'frame_native.py',
            'frame_asset.py', 'native.py', 'mesh.py', 'cover_mesh.py', 'parameters.json',
            'station_coordinates.py', 'episode_checks.py', 'outcome_native.py', 'score_logic.py',
            'score_full_run.py', 'evaluation_contract.py', 'dream_evaluation.py', 'runtime_checks.py',
            'gpu_coordination.py', 'frame_solver.py', 'cloth_solver.py', 'robot_contacts.py',
            'chunk_writer.py', 'measurements.py', 'audit_frame_robot.py', 'audit_cover_parallel.py',
            'audit_full_cover.py', 'full_observation.py', 'full_sdk.py', 'sdk_session.py', 'sdk_yam.py',
            'sdk_cameras.py', 'sdk_cooked_geometry.py', 'read_native_properties.py',
            'handle_held_finish.py', 'score_handle_held.py')
        for file in protected:
            assert sha256(source / file) == manifest[file], 'Validation source changed: ' + file
        assert json.loads((run / 'reset-verification.json').read_text())['passed']
        if key == 'interface_evidence':
            check = json.loads((run / 'interface-verification.json').read_text())
            assert req.get('policy') == 'interface-check' and check['passed']
            assert check['hosted_calls'] == 0 and check['native_execution']
        else:
            score = json.loads((run / result_file).read_text())
            if revision == 'handle-held-v2':
                assert score['rubric_revision'] == revision and not score['historical_screen_only']
                assert sha256(run / 'benchmark-result.json') == score['original_assessment_sha256']
                assert sha256(run / 'outcomes.jsonl') == score['original_observations_sha256']
                original = json.loads((run / 'benchmark-result.json').read_text())
                assert not original['invalid_reasons'] and original['audit_sha256'] == score['audit_sha256']
                for file, digest in score['source_sha256'].items():
                    assert manifest[file] == digest, 'Revised scorer differs from archived sources: ' + file
            assert score['score'] == 1. and score['success'] and not score['invalid_reasons']
            assert sha256(run / observations_file) == score['observations_sha256']
            for artifact, digest in score['audit_sha256'].items():
                assert sha256(run / artifact) == digest
                audit = json.loads((run / artifact).read_text())
                for path, expected_digest in audit['source_sha256'].items():
                    assert sha256(run / path) == expected_digest, 'Stale audit: ' + path
            if key == 'replay_evidence':
                replay = json.loads((run / 'replay-provenance.json').read_text())
                assert replay['source_run'] == request['demo_run'] and replay['fresh_simulation']
                assert replay['from_reset'] and not replay['object_state_writes_during_replay']
                assert replay['actions_sha256'] == sha256(OUT / request['demo_run'] / 'steps.jsonl')
            else:
                assert req.get('policy', 'scripted') == 'scripted' and not req.get('replay_run')
        verified[key] = dict(run=name, manifest_sha256=sha256(run / 'manifest.json'))
    assert credential_status()['api_key_present'], 'Configure DREAM_API_KEY or ~/.dreamlabs/credentials'
    return verified


class TracedSession:
    """Record exactly what the installed integration sends and returns, without secrets."""

    def __init__(self, delegate, out, hosted):
        self.delegate = delegate
        self.out = Path(out)
        self.hosted = hosted
        self.observations = gzip.open(self.out / 'model-observations.jsonl.gz', 'wt')
        self.replies = (self.out / 'inference.jsonl').open('w', buffering=1)
        self.ticks = self.requests = 0

    def check(self, episode):
        self.delegate.check(episode)

    def start(self, episode):
        assert self.ticks == 0 and self.requests == 0, 'Only one episode is authorized'
        self.delegate.start(episode)

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS)
        assert timestamp == self.ticks / CONTROL_HZ
        self.observations.write(json.dumps(dict(step=self.ticks, simulation_time_s=timestamp,
            images_jpeg_base64=dict(frames), joint_pos=vector(state).tolist(), instruction=prompt)) + '\n')
        self.observations.flush()
        self.delegate.observe(frames, vector(state), prompt, timestamp)
        self.ticks += 1

    def act(self, commanded):
        began = time.monotonic()
        chunk = self.delegate.act(commanded)
        self.replies.write(json.dumps(dict(request=self.requests, after_observation_step=self.ticks - 1,
            hosted=self.hosted, elapsed_wall_s=time.monotonic() - began, **chunk.to_json())) + '\n')
        self.requests += 1
        return chunk

    def close(self):
        try:
            self.delegate.close()
        finally:
            self.observations.close()
            self.replies.close()


class OfflineInterfaceSession:
    """No network: small, distinct ordinary targets exercise all fourteen action slots."""

    def __init__(self):
        self.state = None
        self.observed = 0
        self.started = False

    def check(self, episode):
        assert episode.model == MODEL

    def start(self, episode):
        assert not self.started
        self.started = True

    def observe(self, frames, state, prompt, timestamp):
        assert self.started and set(frames) == set(CAMERAS)
        self.state = vector(state)
        self.observed += 1

    def act(self, commanded):
        from inspect_robots_dream.session import Chunk
        assert self.state is not None
        rows = []
        for index in range(48):
            target = self.state.copy()
            target += .008 * np.sin((index + 1) / 16) * np.linspace(.4, 1., 14)
            target[[6, 13]] = self.state[[6, 13]] - .08 * np.sin((index + 1) / 16)
            rows.append(process_action(target))
        return Chunk(rows=np.asarray(rows), rtt_s=0., attempts=0, session_id='offline-interface-check')

    def close(self):
        pass


def run_policy(session):
    """Use the same scene/recorder as the script, with no access to its controller."""
    from inspect_robots import Scene
    from inspect_robots_dream.policy import DreamPolicy
    from inspect_robots_dream.session import DreamSession
    from PIL import Image
    request, out = session.request, Path(session.out)
    mode = request['policy']
    assert mode in ('dream', 'interface-check')
    assert not request.get('replay_run') and not request.get('release_probe_run')
    assert not session.rows and session.physics_steps == 0, 'Begin from the shared declared reset'
    source = Path(__file__).resolve().parent
    revision = validate_task_revision(request, source)
    instruction = json.loads((source / 'task.json').read_text())['instruction']
    hosted = mode == 'dream'
    budget = ACTION_BUDGET if hosted else int(request.get('policy_check_steps', 64))
    assert (hosted and budget == ACTION_BUDGET) or (not hosted and 40 <= budget <= 80)
    evidence = require_hosted_evidence(request, source) if hosted else None
    claim = OUT / 'hosted-dream-attempt.json'
    if hosted:
        # Exclusive creation makes a restarted launcher fail before another paid
        # episode. An error or a zero score never triggers an automatic retry.
        with claim.open('x') as stream:
            json.dump(dict(status='starting', run=out.name, model=MODEL, seed=request['seed'],
                           authorized_episodes=1, evidence=evidence), stream, indent=2)
    traced = TracedSession(DreamSession() if hosted else OfflineInterfaceSession(), out, hosted)
    policy = DreamPolicy(model=MODEL, session=traced, record=False, dry_run=False)
    initial, last = None, None
    changed_cameras = {name: False for name in CAMERAS}
    maximum_joint_change = 0.
    completed = False
    try:
        policy.bind(embodiment_info())
        policy.reset(Scene(id='pocket-mop-cover', instruction=instruction))
        with (out / 'policy-actions.jsonl').open('w', buffering=1) as actions:
            for step in range(budget):
                images, measured = session.public_snapshot()
                observation = public_observation(images, measured, instruction, step)
                if initial is None:
                    initial = observation
                    Image.fromarray(images['top_cam']).save(out / 'setup.png')
                before = session.physics_steps
                began = time.monotonic()
                chunk = policy.act(observation)
                inference_wait = time.monotonic() - began
                assert session.physics_steps == before, 'Physics advanced while waiting for inference'
                assert len(chunk.actions) == 1 and chunk.control_hz == CONTROL_HZ
                assert chunk.terminal is None and chunk.actions[0].terminal is None
                raw = vector(chunk.actions[0].data)
                executed = process_action(raw)
                session.step(raw, 'dream' if hosted else 'offline_interface_check')
                assert session.physics_steps - before == request['physics_hz'] // CONTROL_HZ
                assert len(session.rows) == step + 1
                images, actual = session.public_snapshot()
                last = public_observation(images, actual, instruction, step + 1)
                maximum_joint_change = max(maximum_joint_change, float(abs(last.state['joint_pos'] - initial.state['joint_pos']).max()))
                for name in CAMERAS:
                    changed_cameras[name] |= not np.array_equal(initial.images[name], last.images[name])
                actions.write(json.dumps(dict(step=step, simulation_time_s=(step + 1) / CONTROL_HZ,
                    requested_action=raw.tolist(), executed_stock_target=executed.tolist(),
                    measured_joint_pos=vector(actual).tolist(), inference_wait_wall_s=inference_wait,
                    native_steps=session.physics_steps)) + '\n')
        completed = True
        if not hosted:
            checks = dict(action_count=len(session.rows) == budget,
                native_timing=session.physics_steps == budget * (request['physics_hz'] // CONTROL_HZ),
                preroll_included=policy.preroll_steps == 32,
                all_camera_streams_change=all(changed_cameras.values()),
                measured_motion=maximum_joint_change > 1e-4,
                one_mock_chunk=traced.requests == 1,
                public_fields_only=set(last.state) == {'joint_pos'} and not last.extra,
                complete_recording=len(session.rows) == traced.ticks)
            report = dict(passed=all(checks.values()), checks=checks, native_execution=True,
                hosted_calls=0, diagnostic_only=True, control_steps=budget,
                maximum_measured_state_change=maximum_joint_change, changed_cameras=changed_cameras,
                configuration_sha256=configuration_identity(request),
                joint_labels=list(embodiment_info().action_space.semantics.dim_labels),
                camera_resolution=[WIDTH, HEIGHT], control_hz=CONTROL_HZ,
                rubric_revision=revision, frame_seed=request['seed'], instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                scope='Local native adapter check with installed DreamPolicy and a no-network mock Session; physics task success is not certified.')
            (out / 'interface-verification.json').write_text(json.dumps(report, indent=2))
            assert report['passed'], 'Native public interface check failed'
        return dict(status='completed', termination_reason='action_budget' if hosted else 'interface_check_complete',
                    diagnostic_only=not hosted, hosted_model_attempt=hosted, model=MODEL if hosted else None,
                    control_steps=len(session.rows), physics_steps=session.physics_steps,
                    rubric_revision=revision, scripted_assistance=False, benchmark_score_pending_independent_audit=True)
    finally:
        (out / 'policy-transcript.json').write_text(json.dumps(dict(**policy.transcript(),
            hosted=hosted, completed=completed, received_chunks=traced.requests,
            observation_contract='Instruction, three 320x176 RGB streams and measured left-then-right 14D joints/grippers only.',
            scripted_assistance=False, privileged_data_in_policy=False), indent=2))
        policy.close()
        if hosted:
            claim.write_text(json.dumps(dict(status='episode_completed' if completed else 'episode_error',
                run=out.name, model=MODEL, seed=request['seed'], authorized_episodes=1,
                received_chunks=traced.requests, evidence=evidence,
                repeat_authorized=False, score='pending independent full native audit'), indent=2))
