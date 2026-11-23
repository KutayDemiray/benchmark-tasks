"""Freeze task sources and hash explicit completed evidence without claiming success."""

import argparse
import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from bench import write
from task_paths import OUTPUT_ROOT

HERE=Path(__file__).resolve().parent


def file_record(path):
    """Reject an evidence file that changes while its digest is being computed."""
    before=path.stat()
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), ('Evidence changed during hashing',str(path))
    return dict(sha256=digest,bytes=after.st_size)


def model_activity(run):
    """Summarize actual hosted request evidence, preserving zero and failed attempts."""
    def optional(name):
        path=run/name
        return json.loads(path.read_text()) if path.exists() else {}
    evaluation=optional('evaluation.json')
    model=optional('model-evaluation.json')
    manifest=optional('manifest.json')
    live=(evaluation.get('mode')=='dream_live' or model.get('mode')=='dream_live'
          or any(Path(str(arg)).name=='dream_live_runner.py' for arg in manifest.get('command',[])))
    if not live:
        return None
    path=run/'inference.jsonl'
    events=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    started=sum(event.get('event')=='started' for event in events)
    completed=sum(event.get('event')=='completed' for event in events)
    assert 0<=completed<=started, 'Completed inference lacks a started request'
    if model:
        assert started==model['hosted_requests_started']
        assert completed==model['hosted_requests_completed']
    failure=optional('failure.json')
    if 'hosted_requests_started' in failure:
        assert started==failure['hosted_requests_started']
    result=model.get('result') or optional('worker-00/episode-00/result.json')
    return dict(path=str(run),model=model.get('model'),inference_requests_started=started,
        inference_requests_completed=completed,model_attempt=started>0,
        task_score=result.get('task_score'),executed_control_steps=result.get('control_steps'),
        termination_reason=result.get('termination_reason'),error=model.get('error') or failure or None,
        independently_audited=False,
        scope='Request activity and recorded outcome only; independent scoring and model audit are separate reviews.')


def collect_model_activity(records):
    """Include hosted activity from nested completed run trees exactly once per path."""
    found={}
    def visit(record):
        activity=record.get('model_activity')
        if activity is not None:
            found[activity['path']]=activity
        for child in record.get('children',[]):
            visit(child)
    for record in records:
        visit(record)
    return list(found.values())


def run_record(run,checkpoint):
    """Include completed children; do not hash a live worker's changing trace."""
    complete=(run/'exit.json').is_file()
    assert complete or checkpoint, ('Run still active; use an explicit checkpoint',str(run))
    record=dict(path=str(run.relative_to(OUTPUT_ROOT)),process_completed=complete,evidence_files={})
    for name in ('manifest.json','results.json','exit.json','failure.json','verification.json',
                 'stability-progress.json','coordination-state.json','evaluation-progress.json'):
        if (run/name).exists():record[name.removesuffix('.json')]=json.loads((run/name).read_text())
    if complete:
        paths=list(run.glob('*.json'))+list(run.glob('*.log'))+list(run.glob('*.jsonl'))
        for episode in sorted(run.glob('episode-*')):
            paths.extend(path for path in episode.iterdir() if path.is_file())
        for path in sorted(set(paths)):
            record['evidence_files'][str(path.relative_to(run))]=file_record(path)
        manifest=record.get('manifest',{})
        for name,expected in manifest.get('sources',{}).items():
            assert file_record(run/'task-bundle'/name)['sha256']==expected, ('Frozen source modified',str(run),name)
    else:
        record['pending_note']='Live trace is excluded; JSON summaries above are an as-of snapshot, not completed evidence.'
    children=sorted(path.parent for path in run.glob('*/manifest.json') if path.parent.name not in ('task-bundle','runtime'))
    record['children']=[run_record(child,checkpoint) for child in children]
    record['model_activity']=model_activity(run) if complete else None
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',type=Path,action='append',default=[])
    parser.add_argument('--review',type=Path,action='append',default=[])
    parser.add_argument('--checkpoint',action='store_true',help='Explicitly label active or pending validation as incomplete')
    args=parser.parse_args();out=args.out.resolve()
    args.run=[path.resolve() for path in args.run];args.review=[path.resolve() for path in args.review]
    inputs=[p.resolve() for p in args.run+args.review]
    assert all(p.is_relative_to(OUTPUT_ROOT) for p in [out,*inputs])
    assert not out.exists(), 'Use a fresh bundle directory'
    for run in args.run:
        assert (run/'exit.json').exists() or args.checkpoint, ('Run still active',str(run))
    out.mkdir(parents=True)
    source=out/'task-bundle';source.mkdir()
    hashes={}
    for path in sorted(HERE.iterdir()):
        if not path.is_file():continue
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        shutil.copy2(path,source/path.name)
        hashes[path.name]=file_record(source/path.name)['sha256']
    bundle_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    checks=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(source),'-p','test_*.py'],
        text=True,capture_output=True,check=False)
    (out/'cpu-tests.log').write_text(checks.stdout+checks.stderr)
    environment=json.loads((OUTPUT_ROOT/'environment.json').read_text())
    preservation={}
    for name,expected in environment['read_only_yam_source_hashes'].items():
        actual=file_record(Path(name))['sha256']
        preservation[name]=dict(expected_sha256=expected,actual_sha256=actual,unchanged=actual==expected)
    reviews=[]
    for path in args.review:
        item=dict(path=str(path.relative_to(OUTPUT_ROOT)),**file_record(path))
        if path.suffix=='.json':item['record']=json.loads(path.read_text())
        reviews.append(item)
    runs=[run_record(run,args.checkpoint) for run in args.run]
    model_runs=collect_model_activity(runs)
    report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='incomplete_checkpoint' if args.checkpoint else 'completed_evidence_bundle',
        note='Packaging is not a claim of task success, bimanual benefit, model evaluation or submission readiness. '
             'Consult each native outcome and its independent audit. Large evidence files stay in their original run directories.',
        source_bundle_sha256=bundle_hash,source_hashes=hashes,python_parse_passed=True,
        cpu_test_returncode=checks.returncode,installed_yam_source_preservation=preservation,
        definition=json.loads((source/'task.json').read_text()),environment=environment,
        runs=runs,reviews=reviews,model_runs=model_runs,
        model_calls_performed=any(item['model_attempt'] for item in model_runs),
        uploads_performed=False,submission_performed=False)
    report['packaging_checks_passed']=checks.returncode==0 and all(r['unchanged'] for r in preservation.values())
    write(out/'validation-results.json',report)
    archive=out/'snap-front-shirt-source.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for path in sorted(source.iterdir()):
                info=tar.gettarinfo(str(path),arcname='snap-front-shirt/'+path.name)
                info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
                with path.open('rb') as stream:tar.addfile(info,stream)
    with tarfile.open(archive,'r:gz') as tar:
        restored={}
        for item in tar.getmembers():
            assert item.isfile() and item.name.startswith('snap-front-shirt/')
            with tar.extractfile(item) as stream:restored[Path(item.name).name]=hashlib.file_digest(stream,'sha256').hexdigest()
        assert restored==hashes, 'Archive content differs from frozen source'
    write(out/'bundle-hashes.json',dict(source_bundle_sha256=bundle_hash,
        source_archive_sha256=file_record(archive)['sha256'],
        validation_results_sha256=file_record(out/'validation-results.json')['sha256'],
        cpu_tests_sha256=file_record(out/'cpu-tests.log')['sha256']))
    print(json.dumps(dict(output=str(out),source_bundle_sha256=bundle_hash,
        packaging_checks_passed=report['packaging_checks_passed'],status=report['status'])))
    return 0 if report['packaging_checks_passed'] else 1


if __name__=='__main__':sys.exit(main())
