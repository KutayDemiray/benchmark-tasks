"""Continue bounded local validation after the existing passive gate completes."""

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
from physics_provenance import verify_physical_sources
from task_paths import OUTPUT_ROOT, SCHEDULER_COMMAND

HERE=Path(__file__).resolve().parent
SCHEDULER=SCHEDULER_COMMAND
LOCAL_MODES={'hold','bimanual','ablation','scripted','action_replay'}


def launch_command(bundle,out,stability,mode,seconds,seed=100,extra=()):
    """Always reserve one exclusive stable-task slot for the entire launcher tree."""
    assert mode in LOCAL_MODES, 'Only local action modes are authorized'
    return [SCHEDULER,'launch','--task','snap-front-shirt','--mode','exclusive',
            '--gpu','auto','--wait','300','--',sys.executable,'-B',str(bundle/'launch.py'),
            '--stage','evaluation','--out',str(out),'--','--stability-run',str(stability),
            '--mode',mode,'--control-hz','20','--seconds',str(seconds),'--seed',str(seed),*extra]


def stable_gate(review):
    return review.get('verification_pass') is True and review.get('gate_passed') is True


def retained_pilot(result):
    return (not result.get('invalid_reasons') and result.get('pilot_rigid_contact_guard_passed') is True
            and 0 in result.get('retained_pairs',[]))


class Queue:
    def __init__(self,out,stability):
        self.out,self.stability=out,stability
        self.report=dict(pid=os.getpid(),phase='waiting_for_stability',stages=[],pilots=[],
                         local_validation_complete=False,submission_ready=False,
                         model_calls=False,uploads=False,scope='Bounded local validation; no model or submission operations')

    def state(self,phase,**fields):
        self.report.update(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
        write(self.out/'progress.json',self.report)

    def job(self,name,mode,seconds,seed=100,extra=()):
        from verify_tabletop_run import verify
        out=self.out/name
        command=launch_command(HERE,out,self.stability,mode,seconds,seed,extra)
        self.state('running_'+name)
        with (self.out/(name+'-scheduler.log')).open('x') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        item=dict(name=name,mode=mode,path=str(out),command=command,scheduler_returncode=completed.returncode)
        self.report['stages'].append(item)
        self.state('auditing_'+name)
        assert completed.returncode==0, ('Scheduled stage failed or capacity unavailable',name,completed.returncode)
        assert (out/'results.json').exists() and not (out/'failure.json').exists(), ('Missing/failed native result',name)
        assert json.loads((out/'exit.json').read_text())['returncode']==0
        workers=sorted(out.glob('worker-*'))
        assert workers, 'No native workers'
        results=[]
        for worker in workers:
            audit=verify(worker)
            write(worker/'verification.json',audit)
            assert audit['verification_pass'], ('Independent evidence audit failed',str(worker))
            results.append(json.loads((worker/'episode-00/result.json').read_text()))
        item.update(worker_paths=[str(p) for p in workers],results=results,independent_audit_pass=True)
        self.state('completed_'+name)
        return out,results

    def matched_pair(self,revision,seed,label):
        from analyze_bimanual_pair import compare
        supported,results=self.job(label+'-supported','bimanual',60,seed,['--pilot-revision',str(revision)])
        assert len(results)==1
        item=dict(revision=revision,seed=seed,supported=str(supported),retained=retained_pilot(results[0]),
                  qualifies_for_replication=False)
        self.report['pilots'].append(item)
        if not item['retained']:
            self.state('pilot_did_not_retain',last_pilot=item)
            return False
        unsupported,_=self.job(label+'-unsupported','ablation',60,seed,
            ['--replay-episode',str(supported/'worker-00/episode-00')])
        comparison=compare(supported/'worker-00',unsupported/'worker-00')
        write(self.out/(label+'-comparison.json'),comparison)
        item.update(unsupported=str(unsupported),comparison=comparison,
                    qualifies_for_replication=comparison['qualifies_for_replication'])
        self.state('pilot_compared',last_pilot=item)
        return comparison['qualifies_for_replication']

    def continue_after_gate(self):
        from review_stability_round import review
        gate=review(self.stability)
        write(self.out/'stability-verification.json',gate)
        if not stable_gate(gate):
            self.state('stopped_failed_stability_gate',stability=gate)
            return
        manifest=json.loads((self.stability/'manifest.json').read_text())
        proof=verify_physical_sources(HERE,self.stability/'task-bundle',manifest)
        write(self.out/'physical-source-verification.json',proof)
        self.state('stability_gate_passed',stability=gate)
        interface,results=self.job('interface','hold',1,extra=['--episodes','2','--interface-checks'])
        assert len(results)==2 and all(not r['invalid_reasons'] and r['control_steps']==20 for r in results)
        identities=[]
        for worker in sorted(interface.glob('worker-*')):
            checks=json.loads((worker/'interface-checks.json').read_text())
            assert checks['invalid_action_rejected'] and checks['native_physics_steps_advanced']==0
            identities.append(json.loads((worker/'exit.json').read_text())['worker_pid'])
        assert len(set(identities))==2
        selected=None
        for revision in (1,2,3):
            if self.matched_pair(revision,100,f'pilot-r{revision}-seed100'):
                if self.matched_pair(revision,101,f'pilot-r{revision}-seed101'):
                    selected=revision
                    break
        self.report['repeatable_one_snap_bimanual_revision']=selected
        # Revalidate the established three-snap strategy in the selected stable
        # physical configuration and new 20 Hz public action interface.
        baseline,results=self.job('baseline-reproduction','scripted',90,extra=['--episodes','2'])
        assert len(results)==2
        if not all(r['full_task_success'] and r['task_score']==1. and not r['invalid_reasons'] for r in results):
            self.state('stopped_full_task_reproduction_failed')
            return
        from compare_tabletop_runs import compare as compare_reproduction
        reproduction=compare_reproduction(baseline/'worker-00',baseline/'worker-01')
        write(self.out/'reproduction-verification.json',reproduction)
        from check_tabletop_shortcuts import check
        shortcuts=check(baseline/'worker-00')
        write(self.out/'shortcut-verification.json',shortcuts)
        assert reproduction['passed'] and shortcuts['passed']
        media_command=[sys.executable,'-B',str(HERE/'prepare_review_media.py'),
            '--run',str(baseline/'worker-00'),'--out',str(self.out/'validated-review-media')]
        with (self.out/'media-preparation.log').open('x') as log:
            media=subprocess.run(media_command,stdout=log,stderr=subprocess.STDOUT,check=False)
        assert media.returncode==0, 'Validated demonstration media preparation failed'
        self.report['review_media']=str(self.out/'validated-review-media')
        replay,replay_results=self.job('action-replay','action_replay',90,
            extra=['--replay-episode',str(baseline/'worker-00/episode-00')])
        self.report['action_replay_result']=replay_results
        self.state('local_validation_finished',local_validation_complete=True,
            next_step=('Review one-snap bimanual benefit and extend it only if supported; '
                       'prepare final media; obtain separate authorization for the required model attempt.'),
            bimanual_full_task_success=False)
        write(self.out/'results.json',self.report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--wait-hours',type=float,default=12.)
    args=parser.parse_args();out=args.out.resolve();stability=args.stability_run.resolve()
    assert all(p.is_relative_to(OUTPUT_ROOT) for p in (out,stability))
    assert HERE==out/'task-bundle', 'Launch a frozen task-bundle to avoid changing code mid-queue'
    assert 0<args.wait_hours<=24
    queue=Queue(out,stability)
    deadline=time.monotonic()+args.wait_hours*3600
    try:
        while not (stability/'exit.json').exists():
            if time.monotonic()>=deadline:
                queue.state('stopped_stability_wait_expired');return
            queue.state('waiting_for_stability',stability_run=str(stability))
            time.sleep(30.)
        queue.continue_after_gate()
    except BaseException as exc:
        queue.state('stopped_error',error=repr(exc))
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            submission_ready=False,model_calls=False))
        raise
    finally:
        write(out/'queue-exit.json',dict(phase=queue.report['phase'],
            local_validation_complete=queue.report['local_validation_complete'],submission_ready=False))


if __name__=='__main__':main()
