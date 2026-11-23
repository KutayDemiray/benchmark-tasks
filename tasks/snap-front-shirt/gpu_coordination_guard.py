"""Pause only an owned job tree while uncoordinated physical-GPU workloads run."""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from task_paths import OUTPUT_ROOT, SCHEDULER_COMMAND


def descendants(parents,root):
    """Return the root's descendants in parent-first order."""
    found=[];known={root}
    while True:
        added=[pid for pid,parent in parents.items() if parent in known and pid not in known]
        if not added:return found
        found.extend(added);known.update(added)


def process_record(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,uid=path.stat().st_uid,state=fields[0],parent=int(fields[1]),
            start_ticks=int(fields[19]),command=(path/'cmdline').read_bytes().split(b'\x00')[:-1])
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(expected,actual):
    return bool(actual is not None and expected['uid']==actual['uid']==os.getuid()
                and expected['start_ticks']==actual['start_ticks'] and expected['command']==actual['command'])


def signal_owned(expected,kind):
    """Refuse a recycled PID or changed ownership before sending a signal."""
    if not same_process(expected,process_record(expected['pid'])):return False
    try:descriptor=os.pidfd_open(expected['pid'],0)
    except ProcessLookupError:return False
    try:
        if not same_process(expected,process_record(expected['pid'])):return False
        signal.pidfd_send_signal(descriptor,kind,None,0)
        return True
    except ProcessLookupError:return False
    finally:os.close(descriptor)


def graphics_entries(table):
    pattern=r'\|\s*(\d+)\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+([CG+]+)\s'
    return [(int(gpu),int(pid),kind) for gpu,pid,kind in re.findall(pattern,table)]


def occupancy(uuid):
    table=subprocess.check_output(['nvidia-smi'],text=True,timeout=15)
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                 '--format=csv,noheader'],text=True,timeout=15)
    compute=set()
    for line in raw.splitlines():
        fields=[v.strip() for v in line.split(',')]
        if len(fields)>=2 and fields[1]==uuid:compute.add(int(fields[0]))
    entries=graphics_entries(table)
    listed={pid for gpu,pid,kind in entries if gpu==0}
    assert compute<=listed, 'Cannot parse the physical GPU process table safely'
    return compute,[(pid,kind) for gpu,pid,kind in entries if gpu==0]



def scheduler_enumeration(uuid, snapshot=None):
    """Allow only tiny inactive graphics contexts verified by the canonical scheduler."""
    if snapshot is None:
        snapshot=json.loads(subprocess.check_output([
            SCHEDULER_COMMAND,'status','--json'],text=True,timeout=15))
    matches=[g for g in snapshot['gpus'] if g['uuid']==uuid]
    assert len(matches)==1, 'Missing/ambiguous physical GPU in scheduler inventory'
    accepted=[]
    for entry in matches[0]['processes']:
        if (entry.get('classification')=='registered_graphics_enumeration'
                and entry['type']=='G' and 0<=entry['memory_mib']<=16 and entry.get('job_id')):
            accepted.append(dict(pid=entry['pid'],type=entry['type'],memory_mib=entry['memory_mib'],
                                 job_id=entry['job_id']))
    return accepted


def foreign_processes(compute, entries, baseline, allowed, enumeration=()):
    """Never exempt compute, unknown graphics or an unverified scheduler context."""
    exempt={entry['pid'] for entry in enumeration}
    graphics={pid for pid,kind in entries if (pid,kind) not in baseline
              and not (kind=='G' and pid in exempt)}
    return sorted((set(compute)|graphics)-set(allowed))


def write_state(run,state):
    temporary=run/'.coordination-state.tmp'
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    temporary.replace(run/'coordination-state.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--launcher-pid',type=int,required=True)
    parser.add_argument('--adopt-paused',type=int,action='append',default=[])
    parser.add_argument('--allow-scheduler-enumeration',action='store_true',
        help='Ignore only canonical runtime-verified inactive graphics allocations <=16 MiB')
    args=parser.parse_args()
    run=args.run.resolve();assert run.is_relative_to(OUTPUT_ROOT)
    root=process_record(args.launcher_pid)
    assert root and root['uid']==os.getuid() and str(run).encode() in root['command']
    assert any(v.endswith(b'/launch.py') for v in root['command'])
    assert not (run/'exit.json').exists(), 'Job already ended'
    preflight=json.loads((run/'preflight.json').read_text())
    uuid=preflight['gpu0'].split(',')[0].strip()
    baseline={(pid,kind) for gpu,pid,kind in graphics_entries(preflight['graphics_and_compute_table'])
              if gpu==0 and kind=='G'}
    paused={};clear_count=0;previous=None
    pause_started=None;paused_total=0.
    for pid in args.adopt_paused:
        info=process_record(pid)
        assert info and info['uid']==os.getuid() and info['state'] in ('T','t')
        assert any(str(run).encode() in value for value in info['command'])
        paused[pid]=info
    events=(run/'coordination-events.jsonl').open('a',buffering=1)
    try:
        while same_process(root,process_record(args.launcher_pid)) and not (run/'exit.json').exists():
            raw=subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True)
            parents={int(pid):int(parent) for pid,parent in (line.split() for line in raw.splitlines())}
            children=descendants(parents,args.launcher_pid)
            allowed={args.launcher_pid,*children}
            errors=[];changed=[];enumeration=[]
            try:
                compute,entries=occupancy(uuid)
                if args.allow_scheduler_enumeration:
                    enumeration=scheduler_enumeration(uuid)
                foreign=foreign_processes(compute,entries,baseline,allowed,enumeration)
            except (OSError,ValueError,KeyError,TypeError,AssertionError,subprocess.SubprocessError) as exc:
                foreign=[];errors=[repr(exc)]
            if foreign or errors:
                clear_count=0
                for pid in children:
                    info=process_record(pid)
                    if info and info['uid']==os.getuid() and info['state'] not in ('T','t','Z'):
                        if signal_owned(info,signal.SIGSTOP):
                            paused[pid]=info;changed.append(dict(pid=pid,signal='SIGSTOP'))
                status='waiting_for_gpu_0'
            else:
                clear_count+=1
                if clear_count>=2:
                    for pid,info in reversed(list(paused.items())):
                        if pid in allowed and signal_owned(info,signal.SIGCONT):
                            changed.append(dict(pid=pid,signal='SIGCONT'))
                        paused.pop(pid)
                status='waiting_for_second_clear_check' if paused else 'running'
            now=time.monotonic()
            if paused and pause_started is None:pause_started=now
            elif not paused and pause_started is not None:
                paused_total+=now-pause_started;pause_started=None
            state=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guard_pid=os.getpid(),
                monotonic=now,paused_total_seconds=paused_total,pause_started_monotonic=pause_started,
                launcher_pid=args.launcher_pid,status=status,physical_gpu_uuid=uuid,
                foreign_gpu_pids=foreign,paused_owned_pids=sorted(paused),errors=errors,
                verified_inactive_enumeration=enumeration,
                other_tasks_signalled=False,lock_owner_unchanged=True)
            write_state(run,state)
            key=(status,tuple(foreign),tuple(sorted(paused)),tuple(errors))
            if key!=previous or changed:
                events.write(json.dumps(dict(**state,signals=changed))+'\n')
                print(json.dumps(dict(status=status,foreign_gpu_pids=foreign,signals=changed)),flush=True)
                previous=key
            time.sleep(5.)
        write_state(run,dict(status='launcher_finished_or_identity_changed',guard_pid=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paused_owned_pids=sorted(paused),
            other_tasks_signalled=False))
    finally:events.close()


if __name__=='__main__':main()
