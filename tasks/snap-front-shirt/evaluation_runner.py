"""Local scripted, hold or fake-session evaluation; no live model mode exists."""

import argparse
import hashlib
import json
from pathlib import Path
import traceback
import time

import numpy as np

from bench import snapshot,write
from fresh_workers import inherited_lock
from local_episode import LocalEpisodeCoordinator,SnapShirtEmbodiment
from network_guard import deny_internet_sockets
from policy_contract import INSTRUCTION
from task_paths import OUTPUT_ROOT

from physics_provenance import PHYSICAL_SOURCES,verify_physical_sources
HERE=Path(__file__).resolve().parent


def selected_configuration(report):
    """Reject incomplete screens and failed horizons before starting native evaluation."""
    assert report.get('gate_passed') is True, 'Stability gate has not passed'
    selected=report['selected_configuration']
    matches=[c for c in report['candidates'] if c['configuration']==selected]
    assert len(matches)==1 and matches[0]['passed'] is True, 'No validated configuration'
    episodes=matches[0]['episodes']
    assert [(e['seed'],e['seconds']) for e in episodes]==[(100,10),(101,10),(100,300)], 'Incomplete stability evidence'
    for e in episodes:
        assert e['passed'] is True and e['worker']['returncode']==0, 'Failed stability episode'
        r=e['result']
        assert r['diagnostic_pass'] is True and not r['invalid_reasons'], 'Invalid stability result'
        assert r['control_steps']==e['seconds']*10 and r['physics_steps_observed']==e['seconds']*960+1, 'Truncated stability episode'
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--mode',choices=('scripted','hold','fake','action_replay','bimanual','ablation'),default='scripted')
    parser.add_argument('--control-hz',type=int,choices=(10,20),default=20)
    parser.add_argument('--seconds',type=float,default=90.)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--replay-episode',type=Path)
    parser.add_argument('--pilot-revision',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--interface-checks',action='store_true')
    args=parser.parse_args()
    assert args.episodes>=1 and 0<args.seconds<=300
    assert not args.interface_checks or args.mode=='hold', 'Interface diagnostics use passive holds'
    inherited_lock()
    deny_internet_sockets()  # Offline coordinator and all children: local IPC remains available.
    assert args.stability_run.resolve().is_relative_to(OUTPUT_ROOT)
    configuration=selected_configuration(json.loads((args.stability_run/'results.json').read_text()))
    from review_stability_round import review
    gate_review=review(args.stability_run)
    write(args.out/'stability-verification.json',gate_review)
    assert gate_review['verification_pass'] and gate_review['gate_passed'], 'Independent stability audit failed'
    manifest=json.loads((args.stability_run/'manifest.json').read_text())
    source_proof=verify_physical_sources(HERE,args.stability_run/'task-bundle',manifest)
    write(args.out/'physical-source-verification.json',source_proof)
    bundle=snapshot(args.out)
    if args.mode=='bimanual':
        from bimanual_preflight import controller_screen
        screen=controller_screen(args.pilot_revision)
        write(args.out/'controller-preflight.json',screen)
        assert screen['screen_passed'], 'CPU controller preflight did not pass'
    controller='baseline' if args.mode=='scripted' else 'none'
    if args.mode=='bimanual':controller=f'bimanual_{args.pilot_revision}'
    coordinator=LocalEpisodeCoordinator(args.out,control_hz=args.control_hz,seconds=args.seconds,
        iterations=configuration['iterations'],pitch=configuration['pitch'],controller=controller,pilot_diagnostics=args.mode in ('bimanual','ablation'))
    policy=adapter=None
    replay=None
    if args.mode in ('action_replay','ablation'):
        assert args.replay_episode is not None
        setup=json.loads((args.replay_episode/'setup.json').read_text())
        assert setup['seed']==args.seed, 'Replay seed differs from its source episode'
        if args.mode=='ablation':
            assert setup.get('controller','').startswith('bimanual_'), 'Ablation requires a supported pilot record'
        source_hz=setup['control_hz']
        assert args.control_hz%source_hz==0, 'Replay cannot discard recorded actions'
        replay=np.repeat(np.load(args.replay_episode/'control-steps.npz')['actions'],round(args.control_hz/source_hz),axis=0)
    write(args.out/'evaluation.json',dict(mode=args.mode,control_hz=args.control_hz,seconds=args.seconds,
        episodes=args.episodes,seed=args.seed,stability_run=str(args.stability_run),
        pilot_revision=args.pilot_revision if args.mode=='bimanual' else None,
        replay_episode=None if args.replay_episode is None else str(args.replay_episode),
        ablation='left arm holds its initial pose; right actions replayed exactly' if args.mode=='ablation' else None,
        configuration=configuration,bundle_sha256=bundle,model_call=False,
        scope='Local evaluation. Fake session is an adapter test, not a model attempt.'))
    try:
        for index in range(args.episodes):
            from inspect_robots import Action,Scene
            scene=Scene(id='snap-front-shirt-tabletop',instruction=INSTRUCTION)
            if args.mode in ('scripted','bimanual'):
                current=coordinator.reset(seed=args.seed,instruction=INSTRUCTION)
            else:
                adapter=SnapShirtEmbodiment(coordinator)
                if args.mode=='fake':
                    from offline_policy import OfflineSession
                    from inspect_robots_dream.policy import DreamPolicy
                    policy=DreamPolicy(model='offline/injected-session',session=OfflineSession(),preroll_steps=0)
                    policy.bind(adapter.info)
                    policy.reset(scene)
                current=adapter.reset(scene,seed=args.seed)
            hold=current.state['joint_pos'].copy()
            if args.interface_checks:
                worker=coordinator.worker
                native_index=worker.last_native_index
                time.sleep(2.)
                repeated=coordinator.observe()
                assert repeated.state_time==0. and np.array_equal(repeated.state['joint_pos'],hold)
                assert worker.last_native_index==native_index
                rejected=False
                try:worker.request(op='step',action=np.full(14,np.nan))
                except RuntimeError as exc:
                    assert '14 finite' in str(exc)
                    rejected=True
                assert rejected, 'Native worker accepted a nonfinite action'
                repeated=coordinator.observe()
                assert repeated.state_time==0. and np.array_equal(repeated.state['joint_pos'],hold)
                assert worker.last_native_index==native_index
                write(worker.out/'interface-checks.json',dict(wait_seconds=2.,observation_time_advanced=0.,
                    native_physics_steps_advanced=0,invalid_action_rejected=True,joints_unchanged=True))
            reason='time_limit'
            for step in range(round(args.seconds*args.control_hz)):
                if args.mode in ('scripted','bimanual'):
                    result=coordinator.script_step()
                elif args.mode=='fake':
                    chunk=policy.act(current)
                    assert len(chunk.actions)==1, 'Installed fake-session adapter changed its one-row contract'
                    result=adapter.step(chunk.actions[0])
                else:
                    if replay is not None and step>=len(replay):
                        reason='replay_exhausted';break
                    action=hold.copy() if replay is None else replay[step].copy()
                    if args.mode=='ablation':action[:7]=hold[:7]
                    result=adapter.step(Action(action))
                current=result.observation
                assert abs(current.state_time-(step+1)/args.control_hz)<1e-8, 'Action clock mismatch'
                if result.terminated or result.truncated:
                    reason=result.termination_reason;break
            completed=coordinator.finish(reason)
            if policy is not None:policy.close();policy=None
            write(args.out/'evaluation-progress.json',coordinator.results)
            print(json.dumps(dict(episode=index,score=completed['task_score'],termination_reason=completed['termination_reason'])),flush=True)
        write(args.out/'results.json',coordinator.results)
    except BaseException as exc:
        write(args.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),full_task_success=False))
        try:
            coordinator.finish('evaluation_error')
        except BaseException as cleanup_error:
            write(args.out/'shutdown-failure.json',dict(error=repr(cleanup_error),traceback=traceback.format_exc()))
        raise
    finally:
        if policy is not None:policy.close()
        coordinator.close()


if __name__=='__main__':main()
