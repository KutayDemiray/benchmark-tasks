"""Validate public metadata and exact inspected media without simulation or network access."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ('title', 'category', 'instruction', 'setup', 'scoring', 'run_limit', 'fidelity', 'scores', 'media')
CATEGORIES = ('Impact on jobs', 'Impact on AI timelines', 'Safety and alignment', 'Societal relatability', 'Other')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-missing-media', action='store_true', help='Preparation only; never qualifies for commit/submission')
    parser.add_argument('--media-dir', type=Path, help='Verify the three previously inspected files against recorded hashes')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    task = json.loads((HERE / 'task.json').read_text())
    evidence = json.loads((HERE / 'submission-evidence.json').read_text())
    lengths = {name: len(value) for name, value in task.items() if isinstance(value, str)}
    print(json.dumps({'actual_field_lengths': lengths}, indent=2))
    for name in REQUIRED:
        if name == 'media' and args.allow_missing_media and task.get(name) is None:
            continue
        assert isinstance(task.get(name), str) and task[name].strip(), f'Missing or empty required field: {name}'
    assert task['category'] in CATEGORIES
    assert lengths['fidelity'] <= 500, 'fidelity exceeds 500 characters'
    assert lengths['scores'] <= 300, 'scores exceeds 300 characters'
    assert 'Dream-1: 0.0' in task['scores'] and 'Scripted demo: 1.0' in task['scores']
    assert evidence['model_result']['task_score'] == 0.0
    assert evidence['scripted_demonstration']['score'] == 1.0
    assert evidence['model_attempt']['hosted_requests_started'] == 2
    assert evidence['model_attempt']['hosted_requests_completed'] == 1
    assert evidence['model_attempt']['episode_retries'] == 0
    assert evidence['model_attempt']['scripted_assistance'] is False
    assert evidence['model_attempt']['error']['type'] == 'DreamSessionExpired'
    profile = task['validation_basis']['active_completion']
    assert (profile['control_hz'], profile['action_dimension'], profile['run_limit_seconds']) == (20, 14, 300)
    assert profile['required_snap_indices'] == [0, 1, 2]
    assert '6,000' in task['run_limit'] and '20 Hz' in task['run_limit']
    for name in REQUIRED:
        value = task.get(name)
        if isinstance(value, str):
            assert all(word not in value.lower() for word in ('/home/', '/users/', 'kutay', 'robocurve_token')), name
    checked = {}
    if args.media_dir:
        for name, recorded in evidence['media'].items():
            path = args.media_dir / name
            assert path.is_file(), path
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            size = path.stat().st_size
            assert digest == recorded['sha256'] and size == recorded['bytes'], f'Changed inspected media: {name}'
            assert 0 < size < (10_000_000 if name.endswith('.png') else 50_000_000)
            checked[name] = {'sha256': digest, 'bytes': size}
    report = {'metadata_valid': True, 'media_id_present': bool(task.get('media')),
              'preparation_only': args.allow_missing_media, 'field_lengths': lengths,
              'model_score': 0.0, 'demo_score': 1.0, 'inspected_media_hashes': checked,
              'note': 'Local structure/evidence validation only; not upload authentication, receipt or acceptance.'}
    if args.out:
        assert not args.out.exists(), 'Preserve prior validation records; use a fresh output path'
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
