"""Maintain a lightweight live handoff and audit a finished local queue once."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from task_paths import OUTPUT_ROOT

HERE = Path(__file__).resolve().parent


def read_optional(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return dict(read_error=repr(exc))


def atomic_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def latest_control(stability):
    """Read only complete JSON lines from a bounded tail of the newest trace."""
    paths = list(stability.glob('*/episode-00/trace.jsonl'))
    if not paths:
        return None
    path = max(paths,key=lambda p:p.stat().st_mtime_ns)
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        state = row['physics']
        return dict(trace=str(path),completed_control_time_s=row['time_s'],
            last_control_metrics={key:state[key] for key in
                ('strain','table_penetration_m','table_coverage_fraction','rivet_error_m')},
            note='Latest control sample only; not whole-episode maxima or a gate decision')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--stability-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wait-hours',type=float,default=24.)
    args = parser.parse_args()
    queue,stability,out = (p.resolve() for p in (args.queue,args.stability_run,args.out))
    assert all(p.is_relative_to(OUTPUT_ROOT) for p in (queue,stability,out))
    assert HERE == out/'task-bundle', 'Use a frozen observer bundle'
    assert 0 < args.wait_hours <= 24
    deadline = time.monotonic()+args.wait_hours*3600
    state = {}
    while True:
        progress = read_optional(queue/'progress.json')
        state = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            watcher_pid=os.getpid(),queue=str(queue),queue_progress=progress,
            queue_exit=read_optional(queue/'queue-exit.json'),queue_failure=read_optional(queue/'failure.json'),
            coordination=read_optional(stability/'coordination-state.json'),
            latest_passive_control=latest_control(stability),
            submission_ready=False,native_processes_started=False,processes_signalled=False,
            model_calls=False,uploads=False)
        atomic_write(out/'live-state.json',state)
        if state['queue_exit'] is not None and 'read_error' not in state['queue_exit']:
            command = [sys.executable,'-B',str(HERE/'submission_preflight.py'),
                '--queue',str(queue),'--stability-run',str(stability),'--out',str(out/'final-audit.json')]
            with (out/'final-audit.log').open('x') as log:
                finished = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            state['watcher_status'] = 'queue_finished_audit_recorded' if finished.returncode == 0 else 'audit_error'
            state['audit_returncode'] = finished.returncode
            state['final_audit'] = read_optional(out/'final-audit.json')
            break
        if time.monotonic() >= deadline:
            state['watcher_status'] = 'watch_deadline_reached_queue_untouched'
            break
        time.sleep(30.)
    state['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_write(out/'live-state.json',state)
    atomic_write(out/'exit.json',dict(status=state['watcher_status'],submission_ready=False,
                                    queue_untouched=True,utc=state['utc']))


if __name__ == '__main__':
    main()
