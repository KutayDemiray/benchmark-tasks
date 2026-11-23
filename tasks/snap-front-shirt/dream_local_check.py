"""Native five-second installed Dream-adapter test with all Internet sockets denied."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench import snapshot, write
from dream_execution import execute
from dream_readiness import readiness
from gpu_runtime import inherited_fds, scheduled_assignment
from local_episode import LocalEpisodeCoordinator
from network_guard import deny_internet_sockets
from offline_policy import OfflineSession
from task_paths import OUTPUT_ROOT


class ProbeSession(OfflineSession):
    """Use small ordinary joint targets to verify the real left/right action path."""

    def __init__(self, coordinator):
        super().__init__()
        self.coordinator=coordinator
        self.native_script_rejected=False

    def start(self, episode):
        super().start(episode)
        before=self.coordinator.observe()
        native=self.coordinator.worker.last_native_index
        try:
            self.coordinator.worker.request(op='script_step')
        except RuntimeError as exc:
            assert 'Scripted assistance is disabled' in str(exc)
            self.native_script_rejected=True
        assert self.native_script_rejected
        after=self.coordinator.observe()
        assert self.coordinator.worker.last_native_index==native and after.state_time==0.
        assert np.array_equal(before.state['joint_pos'],after.state['joint_pos'])
        self.initial=after.state['joint_pos'].copy()
        self.rows=np.repeat(self.initial[None],96,axis=0)
        self.rows[:48,0]+=.01
        self.rows[:48,7]-=.01


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--validation-queue',type=Path,required=True)
    parser.add_argument('--demo-run',type=Path,required=True)
    args=parser.parse_args()
    assert all(p.resolve().is_relative_to(OUTPUT_ROOT) for p in (args.out,args.stability_run,args.validation_queue,args.demo_run))
    assert scheduled_assignment() is not None
    inherited_fds()
    deny_internet_sockets()
    ready=readiness(args.stability_run,args.validation_queue,args.demo_run)
    write(args.out/'readiness.json',ready)
    bundle=snapshot(args.out)
    config=ready['configuration']
    coordinator=LocalEpisodeCoordinator(args.out,control_hz=20,seconds=5.,iterations=config['iterations'],
        pitch=config['pitch'],controller='none')
    session=ProbeSession(coordinator)
    try:
        evidence=execute(coordinator,args.out,session,model='offline/injected-session',seconds=5.,seed=ready['demonstration']['seed'])
        write(args.out/'local-evaluation.json',evidence)
        write(args.out/'results.json',[evidence['result']])
        result=evidence['result']
        assert evidence['error'] is None and result['control_steps']==100 and not result['invalid_reasons']
        assert result['task_score']==0. and session.requests==2 and session.observations==99
        startup=evidence['startup_camera_filter']
        assert startup['discarded_observations']==1 and startup['first_forwarded_time_s']==.05
        assert startup['additional_actions']==0 and startup['forwarded_observation_calls']==99
        rows=[json.loads(line) for line in (args.out/'worker-00/episode-00/trace.jsonl').read_text().splitlines()]
        policy=[json.loads(line) for line in (args.out/'policy-steps.jsonl').read_text().splitlines()]
        target_errors=[]
        for native,public in zip(rows,policy,strict=True):
            target_errors.append(float(np.max(np.abs(np.asarray(native['robot_action'])-public['requested_action']))))
        assert max(target_errors)<1e-6, 'Action ordering/encoding differs from the native execution'
        achieved=np.asarray(rows[79]['joint_positions'])-session.initial
        assert achieved[0]>.001 and achieved[7]<-.001, 'Left/right test targets did not move the expected joints'
        write(args.out/'contract-checks.json',dict(passed=True,hosted_calls=0,executed_steps=100,
            requests=session.requests,native_script_rejected=session.native_script_rejected,
            native_step_during_inference=0,action_order='left j0..j5, left gripper, right j0..j5, right gripper',
            maximum_action_encoding_error=max(target_errors),achieved_joint_delta=achieved.tolist(),
            source_bundle_sha256=bundle,not_a_model_attempt=True,startup_camera_filter=startup))
    except BaseException as exc:
        write(args.out/'failure.json',dict(error=repr(exc),hosted_calls=0,full_task_success=False))
        raise
    finally:
        coordinator.close()


if __name__=='__main__':main()
