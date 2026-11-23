"""Audit local evidence and freeze a source-only reproduction bundle."""

from task_paths import OUTPUT_ROOT, SOURCE_CHECKOUT_ROOT

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from audit import audit

HERE = Path(__file__).resolve().parent
OUTPUT = OUTPUT_ROOT


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-prefix', default='', help='Limit evidence audits to this run-name prefix; prior gate review is still included')
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(OUTPUT)
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'task-bundle'
    source.mkdir()
    source_hashes = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file():
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=str(path))
            shutil.copy2(path, source / path.name)
            source_hashes[path.name] = digest(path)
    bundle_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()

    checks = subprocess.run(
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(source), '-p', 'test_*.py', '-v'],
        text=True, capture_output=True, check=False,
    )
    (out / 'scorer-tests.log').write_text(checks.stdout + checks.stderr)
    runs = []
    for run in sorted(OUTPUT.iterdir()):
        if not run.name.startswith(args.run_prefix):
            continue
        if not run.is_dir() or not (run / 'manifest.json').is_file():
            continue
        manifest = json.loads((run / 'manifest.json').read_text())
        item = {'run': run.name, 'manifest': manifest, 'simulation_completed': (run / 'results.json').is_file()}
        for filename in ('exit.json', 'failure.json', 'results.json'):
            if (run / filename).exists():
                item[filename.removesuffix('.json')] = json.loads((run / filename).read_text())
        try:
            item['independent_audit'] = audit(run)
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
            item['audit_error'] = repr(exc)
        files = {}
        candidates = list(run.glob('*.json')) + list(run.glob('*.log'))
        for episode in sorted(run.glob('episode-*')):
            for path in episode.iterdir():
                if path.is_file():
                    candidates.append(path)
        for path in sorted(candidates):
            files[str(path.relative_to(run))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        item['evidence_files'] = files
        runs.append(item)

    environment = json.loads((OUTPUT / 'environment.json').read_text())
    final_git = {}
    for name in environment['git_revisions']:
        repository = SOURCE_CHECKOUT_ROOT / name
        final_git[name] = {
            'revision': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
            'status': subprocess.check_output(['git', '-C', str(repository), 'status', '--short'], text=True).strip(),
        }
    preservation = {}
    for path, old_hash in environment['read_only_yam_source_hashes'].items():
        preservation[path] = {'initial_sha256': old_hash, 'final_sha256': digest(Path(path)),
                              'unchanged': digest(Path(path)) == old_hash}
    definition=json.loads((source/'task.json').read_text())
    tabletop=definition.get('validation_basis',{}).get('active_completion',{}).get('scope')=='tabletop_adaptation'
    successful=[];attempts=0;reset_verified=False
    for run in runs:
        outcomes=run.get('results',[])
        verified=run.get('independent_audit',{})
        for index,outcome in enumerate(outcomes):
            if outcome.get('stage')=='tabletop_task':
                attempts+=1
                if outcome.get('full_task_success') and verified.get('verification_pass'):
                    successful.append(dict(run=run['run'],episode=index,seed=outcome['seed'],
                        bundle_sha256=outcome['bundle_sha256']))
        for check in verified.get('same_process_reset_checks',[]):
            reset_index=int(check['episode'].split('-')[-1])
            if any(o.get('full_task_success') for o in outcomes[:reset_index]):
                reset_verified=True
    reproduction={};visual={};shortcuts={};reproduction_shortcuts={}
    for name,target in (('p11-reproduction-review.json',reproduction),('p11-visual-review.json',visual),
                        ('p11-shortcut-review.json',shortcuts),
                        ('p11-reproduction-shortcut-review.json',reproduction_shortcuts)):
        if (OUTPUT/name).exists():target.update(json.loads((OUTPUT/name).read_text()))
    scorer_hashes_match=bool(shortcuts) and bool(reproduction_shortcuts)
    for review in (shortcuts,reproduction_shortcuts):
        hashes=review.get('reviewed_scorer_source_hashes',{})
        scorer_hashes_match &= bool(hashes) and all(source_hashes.get(name)==value
            for name,value in hashes.items())
    reset_support=[]
    for run in runs:
        outcomes=run.get('results',[])
        for index,outcome in enumerate(outcomes):
            if index and outcome.get('case')=='settle' and any(o.get('full_task_success') for o in outcomes[:index]):
                reset_support.append(dict(run=run['run'],episode=index,passed=outcome.get('diagnostic_pass'),
                    invalid_reasons=outcome.get('invalid_reasons',[])))
    results = {
        'status': definition.get('development_status'),
        'development_decision': definition.get('development_decision'),
        'active_task_scope': 'tabletop_adaptation' if tabletop else 'original_hanger_task',
        'original_hanger_task_success': False,
        'source_bundle_sha256': bundle_hash,
        'evidence_run_prefix': args.run_prefix,
        'source_hashes': source_hashes,
        'scorer_test_returncode': checks.returncode,
        'syntax_checks': 'all source Python files parsed successfully',
        'environment': environment,
        'final_git_read_only_snapshot': final_git,
        'yam_source_preservation': preservation,
        'fidelity_target': 'representative adult snap-front shirt; Wrangler is a reference, not an exact replica',
        'exact_sku_measurements_required': False,
        'hardware_measurements': False,
        'hardware_measurement_availability': 'User confirmed none available on 2026-11-20',
        'criteria_basis': {
            'reviewed_on': '2026-11-20',
            'public_source': 'https://github.com/robocurve/benchmark-tasks/blob/main/AGENTS.md#what-makes-a-task-pass',
            'public_success_rate_threshold': None,
            'project_reliability_targets_origin': 'original user brief; not public RoboCurve thresholds',
            'trajectory_limits_origin': 'local diagnostics; not public RoboCurve thresholds',
            'priority': definition.get('validation_basis',{}).get('current_priority'),
            'scope': 'public sources checked; signed-in board unavailable; no model calls or submission authorized',
        },
        'isolated_gate': {
            'passed': json.loads((OUTPUT/'p08-isolated-gate-review.json').read_text())['passed'],
            'status': 'isolated functional gate passed; see archived aggregate review and both p08 verifications',
            'record': json.loads((OUTPUT/'p08-isolated-gate-review.json').read_text()),
            'reasons': [],
            'unresolved_findings': [
                'unguided peel command replay fails unchanged 1 mm / 5 degree local trajectory checks at 960 and 1920 Hz',
                'both peel command replays closed, held and released; relevance of divergence remains unresolved',
                'hardware and fabric parameters are estimates; no hardware calibration is claimed',
            ],
            'intermediate_evidence': ['guided pressing, retention and withdrawal',
                                      'rejected unseated approaches', 'frictional cloth pinch and slip control',
                                      'guided command replay', 'three isolated parameter sensitivity probes',
                                      'reproduced ordinary-action snap press, upper-cloth grasp, coupled lift and release'],
        },
        'acceptance': {
            'complete_from_reset_robot_success': bool(successful),
            'full_task_attempts_across_revisions': attempts,
            'successful_full_task_episodes': successful,
            'full_task_nominal_target': 'one complete demonstration and one fresh-process controller reproduction',
            'full_task_robustness_trials_completed': 0,
            'full_task_robustness_target': 'no statistical campaign required by approved plan',
            'full_task_scorer_validated': bool(successful) and checks.returncode==0
                and shortcuts.get('passed',False) and reproduction_shortcuts.get('passed',False)
                and scorer_hashes_match,
            'fresh_process_controller_reproduction': reproduction.get('passed',False),
            'post_closure_same_process_reset_verified': reset_verified,
            'post_reset_passive_support_diagnostics': reset_support,
            'post_reset_passive_stability_verified': bool(reset_support)
                and all(r['passed'] for r in reset_support),
            'overhead_video_verified': visual.get('top_cam_verified',False),
            'left_wrist_video_verified': visual.get('left_cam_verified',False),
            'right_wrist_video_verified': visual.get('right_cam_verified',False),
            'target_origin': 'approved implementation plan; historical 5/5 and 9/10 superseded',
            'note': 'Only measured complete tabletop_task outcomes count for the adapted task. Isolated and one-snap development diagnostics do not. Original hanger sequence remains incomplete.',
        },
        'controller_reproduction_review': reproduction,
        'visual_review': visual,
        'shortcut_review': shortcuts,
        'reproduction_shortcut_review': reproduction_shortcuts,
        'reviewed_scorer_hashes_match_delivery': scorer_hashes_match,
        'runs': runs,
    }
    write(out / 'validation-results.json', results)
    # Stable source-only archive. Large traces stay in their immutable run folders.
    archive = out / 'snap-front-shirt-source.tar.gz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for path in sorted(source.iterdir()):
                    info = tar.gettarinfo(str(path), arcname='snap-front-shirt/' + path.name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
    write(out / 'bundle-hashes.json', {
        'source_bundle_sha256': bundle_hash,
        'source_archive_sha256': digest(archive),
        'validation_results_sha256': digest(out / 'validation-results.json'),
    })
    print(json.dumps({'source_bundle_sha256': bundle_hash, 'scorer_tests_passed': checks.returncode == 0,
                      'audited_runs': len(runs), 'output': str(out)}))


if __name__ == '__main__':
    main()
