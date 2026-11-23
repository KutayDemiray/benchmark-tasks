"""Prepare, or explicitly apply, a guard-only handover preserving the native job."""

import argparse
import datetime
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys

from gpu_coordination_guard import process_record,same_process,signal_owned,descendants,scheduler_enumeration
from task_paths import OUTPUT_ROOT,GPU_LOCK


def serializable(info):
    return dict(info,command=[arg.decode(errors='replace') for arg in info['command']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--launcher-pid',type=int,required=True)
    parser.add_argument('--guard-pid',type=int,required=True)
    parser.add_argument('--apply',action='store_true',help='Requires authorization to replace the existing guard')
    args=parser.parse_args()
    run=args.run.resolve();out=args.out.resolve()
    assert all(p.is_relative_to(OUTPUT_ROOT) for p in (run,out))
    root=process_record(args.launcher_pid);guard=process_record(args.guard_pid)
    assert root and guard and root['uid']==guard['uid']==os.getuid()
    assert any(p.endswith(b'/launch.py') for p in root['command']) and str(run).encode() in root['command']
    assert any(p.endswith(b'/gpu_coordination_guard.py') for p in guard['command']) and str(run).encode() in guard['command']
    assert not (run/'exit.json').exists(), 'The native job already completed'
    # Verify the original launcher still owns its original exclusive gate.
    expected=GPU_LOCK.stat();held=[]
    for descriptor in (Path('/proc')/str(args.launcher_pid)/'fd').iterdir():
        try:
            actual=descriptor.stat()
            if (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino):
                info=descriptor.parent.with_name('fdinfo').joinpath(descriptor.name).read_text()
                if 'FLOCK' in info and 'WRITE' in info:held.append(int(descriptor.name))
        except FileNotFoundError:continue
    assert held, 'Original exclusive reservation is not held'
    preflight=json.loads((run/'preflight.json').read_text());uuid=preflight['gpu0'].split(',')[0].strip()
    enumeration=scheduler_enumeration(uuid)
    out.mkdir(parents=True,exist_ok=False)
    frozen=out/'task-bundle';frozen.mkdir()
    for name in ('gpu_coordination_guard.py','task_paths.py'):
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    plan=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),apply=args.apply,
        launcher=serializable(root),old_guard=serializable(guard),exclusive_gate_fds=held,
        verified_inactive_enumeration=enumeration,simulation_restart=False,lock_reacquisition=False,
        other_tasks_signalled=False,scope='Replace only this task owned CPU coordination guard; preserve native worker state.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if not args.apply:
        print(json.dumps(plan,indent=2));return
    assert same_process(root,process_record(root['pid'])) and same_process(guard,process_record(guard['pid']))
    descriptor=os.pidfd_open(guard['pid'],0)
    try:
        assert signal_owned(guard,signal.SIGTERM), 'Guard identity changed; nothing else is signalled'
        assert select.select([descriptor],[],[],15)[0], 'Old guard did not exit; do not start a second guard'
    finally:os.close(descriptor)
    assert same_process(root,process_record(root['pid']))
    raw=subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True)
    parents={int(pid):int(parent) for pid,parent in (line.split() for line in raw.splitlines())}
    paused=[]
    for pid in descendants(parents,root['pid']):
        info=process_record(pid)
        if info and info['state'] in ('T','t'):
            assert info['uid']==os.getuid() and any(str(run).encode() in v for v in info['command'])
            paused.append(pid)
    command=[sys.executable,'-B',str(frozen/'gpu_coordination_guard.py'),'--run',str(run),
             '--launcher-pid',str(root['pid']),'--allow-scheduler-enumeration']
    for pid in paused:command.extend(['--adopt-paused',str(pid)])
    with (out/'guard.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True)
    result=dict(old_guard_pid=guard['pid'],new_guard_pid=child.pid,command=command,
        adopted_paused_pids=paused,launcher_pid=root['pid'],simulation_restart=False,lock_reacquisition=False,
        other_tasks_signalled=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
