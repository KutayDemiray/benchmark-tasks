"""Preview or explicitly cancel only an idle, owned legacy validation queue."""

import argparse
import datetime
import json
import os
from pathlib import Path
import select
import signal
import subprocess

from gpu_coordination_guard import process_record, same_process
from task_paths import OUTPUT_ROOT


def idle_evidence(queue):
    """Refuse a queue that has launched a stage, owns children, or passed its wait."""
    job=json.loads((queue/'job.json').read_text())
    progress=json.loads((queue/'progress.json').read_text())
    assert progress['phase']=='waiting_for_stability' and not progress['stages'] and not progress['pilots']
    assert progress['pid']==job['pid']
    identity=process_record(job['pid'])
    assert identity and identity['uid']==os.getuid()
    expected=[part.encode() for part in job['command']]
    assert identity['command']==expected
    assert str(queue/'task-bundle/local_validation_queue.py').encode() in identity['command']
    assert str(queue).encode() in identity['command']
    parents=subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True)
    assert not any(int(parent)==job['pid'] for pid,parent in (line.split() for line in parents.splitlines()))
    stability=Path(progress['stability_run'])
    assert stability.is_relative_to(OUTPUT_ROOT) and not (stability/'exit.json').exists()
    assert not (queue/'queue-exit.json').exists()
    return identity,progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--apply-approved',action='store_true')
    args=parser.parse_args();queue=args.queue.resolve();out=args.out.resolve()
    assert all(path.is_relative_to(OUTPUT_ROOT) for path in (queue,out))
    identity,progress=idle_evidence(queue)
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_pid=identity['pid'],
        identity=dict(identity,command=[p.decode() for p in identity['command']]),
        previous_progress=progress,signal='SIGINT',requires_user_approval=True,
        effect='Stop only the idle old queue; its normal Python finally writes an exit record. Existing p16 preparation then supplies required checks after the untouched stability job.',
        omitted='Optional bimanual experiments and duplicate 90-second-budget demonstrations',
        simulation_restarted=False,guard_changed=False,gpu_lock_changed=False,other_tasks_signalled=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if not args.apply_approved:
        print(json.dumps(plan));return
    descriptor=os.pidfd_open(identity['pid'])
    try:
        current,_=idle_evidence(queue)
        assert same_process(identity,current), 'Queue identity changed'
        signal.pidfd_send_signal(descriptor,signal.SIGINT,None,0)
        assert select.select([descriptor],[],[],15)[0], 'Queue did not exit; no other process will be signalled'
    finally:
        os.close(descriptor)
    assert (queue/'queue-exit.json').exists(), 'Expected native-free queue exit artifact is missing'
    result=dict(target_pid=identity['pid'],user_approved=True,queue_exited=True,
        reason='Superseded optional legacy studies with required Dream preparation',
        queue_exit=json.loads((queue/'queue-exit.json').read_text()),
        simulation_restarted=False,guard_changed=False,gpu_lock_changed=False,other_tasks_signalled=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
