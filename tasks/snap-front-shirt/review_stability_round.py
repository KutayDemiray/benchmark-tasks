"""Independently audit completed passive workers and the bounded gate decision."""

import argparse
import hashlib
import json
from pathlib import Path

from bench import write
from stability_round import CANDIDATES,valid_passive
from task_paths import OUTPUT_ROOT
from verify_tabletop_run import verify


def verified_worker(path):
    """Reuse an audit only when its verifier and raw evidence hashes still match."""
    cached=path/'verification.json'
    source=Path(__file__).with_name('verify_tabletop_run.py')
    verifier_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if cached.exists():
        report=json.loads(cached.read_text())
        evidence=report.get('evidence_file_sha256',{})
        dependencies=report.get('verifier_dependencies_sha256',{})
        valid=bool(evidence) and report.get('verifier_source_sha256')==verifier_hash and bool(dependencies)
        if valid:
            valid=all(hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()==expected
                      for name,expected in dependencies.items())
        if valid:
            for relative,expected in evidence.items():
                with (path/relative).open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                        valid=False;break
        if valid:return report
    report=verify(path);write(cached,report)
    return report


def review(run):
    """Keep execution success, physical gate failure and task success distinct."""
    report=json.loads((run/'results.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    assert json.loads((run/'exit.json').read_text())['returncode']==0, 'Coordinator runtime failure'
    for name,expected in manifest['sources'].items():
        assert hashlib.sha256((run/'task-bundle'/name).read_bytes()).hexdigest()==expected
    assert report['bundle_sha256']==manifest['bundle_sha256']
    checked=[];selected=None;errors=[];pids=[]
    for index,candidate in enumerate(report['candidates']):
        assert index<len(CANDIDATES) and candidate['configuration']==CANDIDATES[index], 'Unplanned numerical candidate'
        episodes=[]
        for item in candidate['episodes']:
            path=Path(item['path'])
            assert path.resolve().is_relative_to(run.resolve())
            try:
                verification=verified_worker(path)
                result=json.loads((path/'episode-00/result.json').read_text())
                setup=json.loads((path/'episode-00/setup.json').read_text())
                cfg=setup['parameters'];setting=candidate['configuration']
                assert verification['verification_pass'], 'Native evidence audit failed'
                assert verification['source_bundle_sha256']==manifest['bundle_sha256']
                assert result==item['result'], 'Coordinator summary differs from native result'
                assert cfg['fabric']['solver_iterations']==setting['iterations']
                assert cfg['garment']['body_mesh_pitch_m']==setting['pitch']
                assert cfg['fabric']['contact_offset_m']==.001 and cfg['simulation']['physics_hz']==960
                assert setup['seed']==item['seed'] and setup['case']=='settle'
                native_pid=json.loads((path/'worker.json').read_text())['worker_pid']
                assert native_pid not in pids, 'Reset reused a native process'
                pids.append(native_pid)
                passed=valid_passive(result,item['seconds'])
                assert passed==item['passed'], 'Gate summary differs from actual diagnostic'
                if passed:
                    assert result['physics_steps_observed']==item['seconds']*960+1
                episodes.append(dict(path=str(path),seed=item['seed'],seconds=item['seconds'],
                    verification_pass=True,passed=passed,worker_pid=native_pid))
            except (AssertionError,KeyError,ValueError,FileNotFoundError) as exc:
                errors.append(dict(path=str(path),error=repr(exc)))
                episodes.append(dict(path=str(path),verification_pass=False,passed=False))
        complete=([(e.get('seed'),e.get('seconds')) for e in episodes]==[(100,10),(101,10),(100,300)]
                  and all(e['passed'] for e in episodes))
        assert bool(candidate['passed'])==complete, 'Candidate selection lacks complete evidence'
        if complete:selected=candidate['configuration']
        checked.append(dict(configuration=candidate['configuration'],episodes=episodes,passed=complete))
    assert selected==report['selected_configuration']
    gate_passed=selected is not None and not errors
    assert bool(report['gate_passed'])==gate_passed
    return dict(scope='Passive numerical validity gate; no task demonstration',
        verification_pass=not errors,gate_passed=gate_passed,selected_configuration=selected,
        candidates=checked,errors=errors,bundle_sha256=manifest['bundle_sha256'],
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full_task_success=False,original_hanger_task_success=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert all(path.resolve().is_relative_to(OUTPUT_ROOT) for path in (args.run,args.out))
    result=review(args.run);write(args.out,result)
    print(json.dumps({k:result[k] for k in ('verification_pass','gate_passed','selected_configuration')}))


if __name__=='__main__':main()
