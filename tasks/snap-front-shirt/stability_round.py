"""Bounded passive screening of three declared numerical/reset configurations."""

import argparse
import json
from pathlib import Path

from bench import snapshot, write
from fresh_workers import inherited_lock, run_worker

CANDIDATES = (
    dict(name='fresh-process', iterations=64, pitch=.01),
    dict(name='cloth-128', iterations=128, pitch=.01),
    dict(name='body-5mm', iterations=64, pitch=.005),
)


def valid_passive(result, seconds, control_hz=10):
    return (result.get('diagnostic_pass') is True and not result.get('invalid_reasons')
            and result.get('termination_reason') == 'diagnostic_completed'
            and result.get('control_steps') == round(seconds*control_hz))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inherited_lock()
    bundle = snapshot(args.out)
    report = dict(bundle_sha256=bundle,candidates=[],selected_configuration=None,
                  full_task_success=False,scope='Passive validity gate, not a task demonstration')
    for candidate in CANDIDATES:
        record = dict(configuration=candidate,episodes=[],passed=False)
        report['candidates'].append(record)
        for label,seed,seconds in (('screen-100',100,10),('screen-101',101,10),('horizon-100',100,300)):
            out = args.out / (candidate['name']+'-'+label)
            status = run_worker(out,args.out,[
                '--case','settle','--seconds',str(seconds),'--seed',str(seed),
                '--cloth-contact-offset','.001','--strain-witness',
                '--cloth-solver-iterations',str(candidate['iterations']),
                '--body-mesh-pitch',str(candidate['pitch'])])
            result_path = out/'episode-00/result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            passed = status['returncode']==0 and valid_passive(result,seconds)
            record['episodes'].append(dict(path=str(out),seed=seed,seconds=seconds,
                                           worker=status,result=result,passed=passed))
            write(args.out/'stability-progress.json',report)
            print(json.dumps(dict(candidate=candidate['name'],episode=label,passed=passed)),flush=True)
            if status['returncode']!=0 or not result:
                report['runtime_failure']=dict(path=str(out),worker=status,
                    note='No numerical candidate is selected from a failed/incomplete native worker.')
                report['gate_passed']=False
                write(args.out/'results.json',report)
                return
            if not passed:
                break
        else:
            record['passed'] = True
            report['selected_configuration'] = candidate
            break
    report['gate_passed'] = report['selected_configuration'] is not None
    write(args.out/'results.json',report)


if __name__ == '__main__':
    main()
