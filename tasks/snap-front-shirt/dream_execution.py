"""Shared ordinary-action loop for installed Dream adapters and local interface tests."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from inspect_robots import Scene
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession, Chunk, DEFAULT_MODEL
from inspect_robots_dream import _station

from bench import write
from local_episode import SnapShirtEmbodiment
from policy_contract import CAMERAS, INSTRUCTION, validate_action


def public_observation(observation, step):
    """Enforce the entire policy input whitelist before it reaches the adapter."""
    assert set(observation.images) == set(CAMERAS)
    assert set(observation.state) == {'joint_pos'} and not observation.extra
    assert observation.instruction == INSTRUCTION
    assert abs(observation.state_time-step/20) < 1e-8
    assert set(observation.image_times) == set(CAMERAS)
    assert all(abs(value-step/20)<1e-8 for value in observation.image_times.values())
    validate_action(observation.state['joint_pos'])
    for name in CAMERAS:
        pixels = observation.images[name]
        assert pixels.dtype == np.uint8 and pixels.shape == (176,320,3)


class StartupCameraFilter:
    """Discard the stale reset image; preserve the installed 32-action preroll."""

    def __init__(self, session):
        self.session = session
        self.seen = 0
        self.forwarded = 0
        self.discarded = 0

    def check(self, episode):
        return self.session.check(episode)

    def start(self, episode):
        assert self.seen == 0, 'Startup filter cannot restart an episode'
        return self.session.start(episode)

    def observe(self, frames, state, prompt, timestamp):
        assert abs(timestamp - self.seen / 20) < 1e-8, 'Nonsequential policy observation'
        step = self.seen
        self.seen += 1
        if step == 0:
            # Native wrist render transforms synchronize after the first ordinary
            # hold already supplied by DreamPolicy. Do not send its stale reset
            # sample, invent a replacement frame, or execute an additional action.
            self.discarded += 1
            return
        self.forwarded += 1
        return self.session.observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        assert self.seen >= 33 and self.discarded == 1, 'Installed preroll must precede inference'
        return self.session.act(commanded)

    def close(self):
        return self.session.close()

    def record(self):
        return dict(discarded_observations=self.discarded, discarded_time_s=0.,
            first_forwarded_time_s=.05 if self.forwarded else None,
            forwarded_observation_calls=self.forwarded, additional_actions=0,
            reason='Native wrist render transforms are stale only at reset; the first standard preroll hold synchronizes them.')


class LoggedSession(DreamSession):
    """Use installed session setup/encoding with one request per inference and no episode retry."""

    def __init__(self, out, *, make_client=None, redact=lambda value:value):
        super().__init__(make_client=make_client)
        self.out = Path(out)
        self.redact = redact
        self.requests_started = 0
        self.requests_completed = 0
        self.starts = 0
        self.events = (self.out/'inference.jsonl').open('x',buffering=1)
        self.observation_events = (self.out/'inference-observations.jsonl').open('x',buffering=1)
        self.observations = 0

    def start(self, episode):
        assert self.starts == 0, 'A hosted episode must not be retried'
        assert episode.model == DEFAULT_MODEL and episode.record is False
        self.starts += 1
        return super().start(episode)

    def observe(self, frames, state, prompt, timestamp):
        """Keep the exact public JPEG/state inputs locally before installed SDK encoding."""
        assert set(frames)==set(CAMERAS)
        measured=validate_action(state)
        record=dict(index=round(timestamp*20),time_s=timestamp,instruction=prompt,
                    joint_pos=measured.tolist(),camera_jpeg_base64=dict(frames))
        self.observation_events.write(json.dumps(record,allow_nan=False)+'\n')
        self.observations += 1
        return super().observe(frames,state,prompt,timestamp)

    def act(self, commanded):
        client = self._open()
        number = self.requests_started
        self.requests_started += 1
        self.events.write(json.dumps(dict(request=number,event='started',monotonic=time.monotonic()))+'\n')
        started = time.monotonic()
        try:
            # No DreamSession retry loop, fallback model, or trajectory repair.
            response = client.get_action(commanded=None if commanded is None else _station.commanded(commanded))
            rows = _station.rows(response)
        except Exception as error:
            self.events.write(json.dumps(dict(request=number,event='error',elapsed_s=time.monotonic()-started,
                error_type=type(error).__name__,error=self.redact(str(error))))+'\n')
            raise
        timing = client.last_timing or {}
        chunk = Chunk(rows=rows,rtt_s=time.monotonic()-started,attempts=1,
                      infer_ms=timing.get('serve_infer_ms'),session_id=client.session_id)
        self.requests_completed += 1
        self.events.write(json.dumps(dict(request=number,event='completed',chunk=chunk.to_json(),timing=timing))+'\n')
        return chunk

    def close(self):
        try:
            super().close()
        finally:
            self.events.close()
            self.observation_events.close()


def execute(coordinator, out, session, *, model=DEFAULT_MODEL, seed=100, seconds=300., redact=lambda value:value):
    """Feed only measured public inputs to Dream and preserve every executed action."""
    assert coordinator.settings['controller'] == 'none'
    assert coordinator.settings['control_hz'] == 20 and 0<seconds<=300
    assert coordinator.settings['seconds'] == seconds
    out = Path(out)
    adapter = SnapShirtEmbodiment(coordinator)
    startup = StartupCameraFilter(session)
    policy = DreamPolicy(model=model,session=startup,rows_per_request=48,record=False,
                         timeout_s=45.,dry_run=False)  # Standard 32-tick observation preroll counts in the budget.
    scene = Scene(id='snap-front-shirt-tabletop',instruction=INSTRUCTION)
    reason = 'time_limit'
    error = None
    current = adapter.reset(scene,seed=seed)
    public_observation(current,0)
    import imageio.v2 as imageio
    for name in CAMERAS:
        imageio.imwrite(out/('initial-'+name+'.png'),current.images[name])
    write(out/'initial-policy-state.json',dict(joint_pos=current.state['joint_pos'].tolist(),
        time_s=0.,camera_order=list(CAMERAS),instruction=INSTRUCTION,policy_context_forwarded=False))
    try:
        policy.bind(adapter.info)
        policy.reset(scene)
        with (out/'policy-steps.jsonl').open('x',buffering=1) as events:
            for step in range(round(seconds*20)):
                public_observation(current,step)
                if step == 1:
                    for name in CAMERAS:
                        imageio.imwrite(out/('first-forwarded-'+name+'.png'),current.images[name])
                    write(out/'first-forwarded-policy-state.json',dict(joint_pos=current.state['joint_pos'].tolist(),
                        time_s=current.state_time,step=step,camera_order=list(CAMERAS),instruction=INSTRUCTION))
                # The native child waits on IPC during inference. Observe before/after
                # verifies that a hosted round trip cannot advance its physical clock.
                before = coordinator.observe()
                before_index = coordinator.worker.last_native_index
                sent = time.monotonic()
                chunk = policy.act(current)
                after = coordinator.observe()
                assert coordinator.worker.last_native_index == before_index
                assert before.state_time == after.state_time
                assert np.array_equal(before.state['joint_pos'],after.state['joint_pos'])
                assert all(np.array_equal(before.images[name],after.images[name]) for name in CAMERAS)
                assert len(chunk.actions) == 1 and chunk.control_hz == 20
                action = chunk.actions[0]
                validate_action(action.data)
                row = dict(step=step,input_time_s=current.state_time,
                    input_joint_pos=current.state['joint_pos'].tolist(),
                    camera_sha256={name:hashlib.sha256(current.images[name].tobytes()).hexdigest() for name in CAMERAS},
                    requested_action=action.data.tolist(),inference_wall_s=time.monotonic()-sent,
                    native_physics_steps_during_inference=0)
                result = adapter.step(action)
                current = result.observation
                public_observation(current,step+1)
                row.update(output_time_s=current.state_time,score=result.reward,
                           terminated=result.terminated,truncated=result.truncated,termination_reason=result.termination_reason)
                events.write(json.dumps(row)+'\n')
                write(out/'policy-transcript.json',policy.transcript())
                if result.terminated or result.truncated:
                    reason = result.termination_reason
                    break
    except Exception as exc:
        reason = 'model_error' if model == DEFAULT_MODEL else 'local_adapter_error'
        error = dict(type=type(exc).__name__,message=redact(str(exc)))
        write(out/'policy-error.json',error)
    finally:
        write(out/'startup-camera-filter.json',startup.record())
        write(out/'policy-transcript.json',policy.transcript())
        try:
            policy.close()
        except Exception as exc:
            write(out/'policy-close-error.json',dict(type=type(exc).__name__,message=redact(str(exc))))
        result = coordinator.finish(reason)
    return dict(result=result,error=error,model=model,seed=seed,control_hz=20,
                instruction=INSTRUCTION,declared_action_budget=6000,executed_steps=result['control_steps'],
                controller='none',scripted_assistance=False,session_recording=False,
                simulation_paused_during_inference=True,startup_camera_filter=startup.record())
