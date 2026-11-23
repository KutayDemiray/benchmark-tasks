"""Bounded binding check using fresh passive workers; not a task-success test."""

import argparse
import json
from pathlib import Path
import traceback

from bench import snapshot, write
from fresh_workers import inherited_lock, run_worker
from gpu_runtime import scheduled_assignment, scheduler_record, audit_record
from task_paths import OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    assert .5 <= args.seconds <= 2.
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    inherited_lock()
    assert scheduled_assignment() is not None, 'This diagnostic requires a scheduler reservation'
    bundle = snapshot(args.out)
    report = dict(scope='Scheduler binding and fresh-process inheritance only; passive task score is zero.',
                  bundle_sha256=bundle, scheduler=scheduler_record(), workers=[], binding_pass=False,
                  full_task_success=False)
    try:
        for index in range(args.workers):
            out = args.out / f'worker-{index:02d}'
            arguments = ['--case', 'settle', '--seconds', str(args.seconds), '--seed', '100',
                         '--cloth-contact-offset', '.001']
            if args.record:
                arguments.append('--record')
            worker = run_worker(out, args.out, arguments)
            assert worker['returncode'] == 0, ('Binding worker failed', out)
            gpu = json.loads((out/'episode-00/gpu.json').read_text())
            audit_record(gpu, json.loads((out/'preflight.json').read_text()), args.record)
            from verify_tabletop_run import verify
            audit = verify(out)
            write(out/'verification.json', audit)
            assert audit['verification_pass'], 'Worker evidence audit failed'
            report['workers'].append(dict(output=str(out), **worker,
                                          device_verification=gpu['binding_verification'],
                                          evidence_audit_pass=audit['verification_pass']))
            write(args.out/'progress.json', report)
        assert len({worker['worker_pid'] for worker in report['workers']}) == args.workers
        report['binding_pass'] = True
        write(args.out/'results.json', report)
    except BaseException as exc:
        write(args.out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                                           partial_report=report, full_task_success=False))
        raise


if __name__ == '__main__':
    main()
