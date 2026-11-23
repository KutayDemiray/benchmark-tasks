"""One genuine hosted Dream-1 episode after independently verified native preparation."""

import argparse
import datetime
import json
import os
from pathlib import Path

from bench import snapshot, write
from task_paths import OUTPUT_ROOT
from submission_preflight import PendingEvidence


def credential_status(out):
    """Read configured credentials without printing, copying or transmitting a secret."""
    os.environ['DREAM_LOG_DIR']=str(out/'dream-logs')
    from dream.credentials import load
    credentials=load()
    return credentials,dict(configured=bool(credentials.api_key),file_exists=credentials.path.is_file(),
        credentials_file=str(credentials.path),required_variable='DREAM_API_KEY',
        gateway=credentials.gateway_url,platform=credentials.platform_url,network_calls=0)


def claim_attempt(path, out):
    """Reserve this authorized episode once, even if a later request fails or crashes."""
    with path.open('x') as stream:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output=str(out),
            status='claimed_before_hosted_access',repeat_automatically=False),stream,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--validation-queue',type=Path,required=True)
    parser.add_argument('--demo-run',type=Path,required=True)
    parser.add_argument('--local-check',type=Path,required=True)
    parser.add_argument('--demo-media',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--attempt-record',type=Path,default=OUTPUT_ROOT/'authorized-dream-attempt-2026-11-22.json')
    parser.add_argument('--check-prerequisites',action='store_true',help='CPU only; no simulator or hosted request')
    args=parser.parse_args()
    for path in (args.out,args.stability_run,args.validation_queue,args.demo_run,args.local_check,args.demo_media,args.attempt_record):
        assert path.resolve().is_relative_to(OUTPUT_ROOT)
    args.out.mkdir(parents=True,exist_ok=args.check_prerequisites or args.out.is_dir())
    credentials,access=credential_status(args.out)
    write(args.out/'credential-status.json',access)
    if args.check_prerequisites:
        from dream_readiness import readiness
        try:
            ready=readiness(args.stability_run,args.validation_queue,args.demo_run,args.local_check)
            from submission_preflight import media_check
            ready['inspected_demo_media']=media_check(args.demo_media,args.demo_run)
            status=dict(native_ready=True,credentials=access,preparation=ready)
        except (AssertionError,FileNotFoundError,KeyError,RuntimeError,PendingEvidence) as exc:
            status=dict(native_ready=False,credentials=access,preparation_error=repr(exc))
        status.update(hosted_calls=0,model_attempt=False)
        write(args.out/'prerequisites.json',status)
        print(json.dumps(dict(native_ready=status['native_ready'],credentials_configured=access['configured'],hosted_calls=0)))
        return 0
    from gpu_runtime import scheduled_assignment, inherited_fds
    assignment=scheduled_assignment()
    assert assignment is not None, 'Every live simulator launch requires the canonical scheduler'
    inherited_fds()
    assert not args.attempt_record.exists(), 'This authorized attempt has already been claimed'
    from dream_readiness import readiness
    ready=readiness(args.stability_run,args.validation_queue,args.demo_run,args.local_check)
    from submission_preflight import media_check
    ready['inspected_demo_media']=media_check(args.demo_media,args.demo_run)
    write(args.out/'readiness.json',ready)
    assert credentials.api_key, 'Missing DREAM_API_KEY and configured credential file; no hosted attempt made'
    assert args.seed==ready['demonstration']['seed'], 'Use the declared demonstration reset seed'
    bundle=snapshot(args.out)
    from local_episode import LocalEpisodeCoordinator
    from dream_execution import LoggedSession, execute
    settings=ready['configuration']
    coordinator=LocalEpisodeCoordinator(args.out,control_hz=20,seconds=300.,iterations=settings['iterations'],
        pitch=settings['pitch'],controller='none')
    redact=lambda text:text.replace(credentials.api_key,'[REDACTED]')
    session=None
    try:
        claim_attempt(args.attempt_record,args.out)
        session=LoggedSession(args.out,redact=redact)
        evidence=execute(coordinator,args.out,session,seed=args.seed,seconds=300.,redact=redact)
        evidence.update(mode='dream_live',source_bundle_sha256=bundle,
            hosted_requests_started=session.requests_started,hosted_requests_completed=session.requests_completed,
            model_attempt=session.requests_started>0,episode_retries=0,model_fallback=False)
        write(args.out/'model-evaluation.json',evidence)
        write(args.out/'evaluation.json',dict(mode='dream_live',control_hz=20,seconds=300.,episodes=1,
            seed=args.seed,model_call=session.requests_started>0,configuration=settings,bundle_sha256=bundle))
        from verify_tabletop_run import verify
        independent=verify(args.out/'worker-00')
        write(args.out/'worker-00/verification.json',independent)
        assert independent['verification_pass']
        assert independent['episodes'][0]['task_score']==evidence['result']['task_score']
        write(args.out/'results.json',[evidence['result']])
    except BaseException as exc:
        write(args.out/'failure.json',dict(type=type(exc).__name__,error=redact(str(exc)),
            hosted_requests_started=0 if session is None else session.requests_started,full_task_success=False))
        raise
    finally:
        coordinator.close()
        if session is not None:session.close()


if __name__=='__main__':main()
