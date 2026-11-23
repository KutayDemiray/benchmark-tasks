"""Finish independent native preparation after existing shirt jobs, never make hosted calls."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bench import write
from task_paths import OUTPUT_ROOT, SCHEDULER_COMMAND

HERE=Path(__file__).resolve().parent
SCHEDULER=SCHEDULER_COMMAND


def command(stage, out, arguments):
    """Wrap the complete adapted launcher tree in one exclusive stable-ID reservation."""
    assert stage in ('evaluation','dream_local_check'), 'This preparation queue cannot run a hosted policy'
    if stage=='evaluation':
        assert arguments[arguments.index('--mode')+1] in ('hold','scripted','action_replay')
    return [SCHEDULER,'launch','--task','snap-front-shirt','--mode','exclusive','--gpu','auto','--wait','300','--',
        sys.executable,'-B',str(HERE/'launch.py'),'--stage',stage,'--out',str(out),'--',*arguments]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--previous-queue',type=Path,required=True)
    parser.add_argument('--wait-hours',type=float,default=24.)
    args=parser.parse_args();out=args.out.resolve();stability=args.stability_run.resolve();previous=args.previous_queue.resolve()
    assert all(path.is_relative_to(OUTPUT_ROOT) for path in (out,stability,previous))
    assert HERE==out/'task-bundle', 'Use a frozen preparation bundle'
    assert 0<args.wait_hours<=24
    report=dict(pid=os.getpid(),phase='waiting_for_existing_jobs',hosted_calls=0,model_attempt=False,stages=[],submission_ready=False)
    def state(phase,**fields):
        report.update(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
        write(out/'progress.json',report)
    def launch(name,stage,arguments):
        destination=out/name
        assert not destination.exists(), 'Never overwrite or silently retry a native stage'
        cmd=command(stage,destination,arguments)
        state('running_'+name)
        with (out/(name+'-scheduler.log')).open('x') as log:
            run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
        report['stages'].append(dict(name=name,command=cmd,returncode=run.returncode))
        state('checking_'+name)
        assert run.returncode==0 and (destination/'results.json').exists() and not (destination/'failure.json').exists()
        assert json.loads((destination/'exit.json').read_text())['returncode']==0
        return destination
    try:
        deadline=time.monotonic()+args.wait_hours*3600
        while not ((stability/'exit.json').exists() and (previous/'queue-exit.json').exists()):
            state('waiting_for_existing_jobs',stability_run=str(stability),previous_queue=str(previous))
            if time.monotonic()>=deadline:
                state('wait_expired_existing_jobs_untouched');return
            time.sleep(30.)
        from review_stability_round import review,verified_worker
        from submission_preflight import interface_check,full_demonstration
        from physics_provenance import verify_physical_sources
        gate=review(stability);write(out/'stability-verification.json',gate)
        assert gate['verification_pass'] and gate['gate_passed'], 'Physical stability gate did not pass'
        verify_physical_sources(HERE,stability/'task-bundle',json.loads((stability/'manifest.json').read_text()))
        common=['--stability-run',str(stability),'--control-hz','20','--seed','100']
        if (previous/'interface').exists():
            validation=previous
        else:
            launch('interface','evaluation',common+['--mode','hold','--seconds','1','--episodes','2','--interface-checks'])
            validation=out
        interface=interface_check(validation);write(out/'interface-verification.json',interface)
        baseline=previous/'baseline-reproduction'
        reusable=False
        if baseline.exists():
            assert (baseline/'exit.json').exists() and json.loads((baseline/'exit.json').read_text())['returncode']==0
            assert not (baseline/'failure.json').exists(), 'Existing baseline failed; inspect it instead of silently retrying'
            for worker in sorted(baseline.glob('worker-*')):
                assert verified_worker(worker)['verification_pass']
                full_demonstration(json.loads((worker/'episode-00/result.json').read_text()),
                    json.loads((worker/'episode-00/setup.json').read_text()),gate['selected_configuration'])
            evaluation=json.loads((baseline/'evaluation.json').read_text())
            reusable=evaluation['mode']=='scripted' and evaluation['control_hz']==20 and evaluation['seconds']==300.
        if not reusable:
            # A successful older 90-second profile does not establish the shared
            # 300-second budget. This is a new declared profile, never a failure retry.
            baseline=launch('baseline-reproduction','evaluation',common+['--mode','scripted','--seconds','300','--episodes','2'])
        workers=sorted(baseline.glob('worker-*'));assert workers
        for worker in workers:
            audit=verified_worker(worker);assert audit['verification_pass']
            setup=json.loads((worker/'episode-00/setup.json').read_text())
            result=json.loads((worker/'episode-00/result.json').read_text())
            full_demonstration(result,setup,gate['selected_configuration'])
        demo=workers[0]
        from dream_readiness import readiness
        write(out/'demonstration-readiness.json',readiness(stability,validation,demo))
        local=launch('dream-local-check','dream_local_check',['--stability-run',str(stability),
            '--validation-queue',str(validation),'--demo-run',str(demo)])
        ready=readiness(stability,validation,demo,local);write(out/'native-readiness.json',ready)
        from check_tabletop_shortcuts import check
        write(out/'shortcut-verification.json',check(demo))
        evidence_queue=baseline.parent
        if evidence_queue==out:
            launch('action-replay','evaluation',common+['--mode','action_replay','--seconds','300',
                '--episodes','1','--replay-episode',str(demo/'episode-00')])
        media=out/'demo-media'
        with (out/'media.log').open('x') as log:
            result=subprocess.run([sys.executable,'-B',str(HERE/'prepare_review_media.py'),
                '--run',str(demo),'--out',str(media)],stdout=log,stderr=subprocess.STDOUT,check=False)
        assert result.returncode==0
        from submission_preflight import audit as preflight
        report['physical_and_interface_checks_complete']=True
        state('checking_final_preflight')
        preflight_report=preflight(evidence_queue,stability,media,interface_queue=validation)
        write(out/'submission-preflight.json',preflight_report)
        required=[item for item in preflight_report['checks'] if item['name']!='selected_demonstration_media']
        assert all(item['status']=='passed' for item in required), 'An independent preparation check failed'
        live=[SCHEDULER,'launch','--task','snap-front-shirt','--mode','exclusive','--gpu','auto','--wait','300','--',
            sys.executable,'-B',str(HERE/'launch.py'),'--stage','dream_live','--out',str(OUTPUT_ROOT/'p16-dream-attempt'),'--',
            '--stability-run',str(stability),'--validation-queue',str(validation),'--demo-run',str(demo),
            '--local-check',str(local),'--demo-media',str(media),'--seed','100']
        state('native_preparation_complete_needs_visual_review_and_credentials',native_preparation_complete=True,
            demonstration=str(demo),local_adapter_check=str(local),demo_media=str(media),live_command=live,
            validation_queue=str(validation),evidence_queue=str(evidence_queue))
        write(out/'results.json',report)
    except BaseException as exc:
        state('stopped_error',error=repr(exc))
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),hosted_calls=0))
        raise
    finally:
        write(out/'queue-exit.json',dict(phase=report['phase'],hosted_calls=0,model_attempt=False,submission_ready=False))


if __name__=='__main__':main()
